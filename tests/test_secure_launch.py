#!/usr/bin/env python3
"""Offline tests for the file-backed SGLang security launcher."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "secure_launch", ROOT / "docker" / "secure-launch.py"
)
assert SPEC and SPEC.loader
secure_launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(secure_launch)


class SecureLaunchTest(unittest.TestCase):
    def _secret(self, directory: Path, name: str, value: str) -> Path:
        path = directory / name
        path.write_text(value + "\n", encoding="ascii")
        path.chmod(0o400)
        return path

    def test_secret_file_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            valid = self._secret(directory, "valid", "a" * 48)
            with mock.patch.dict(os.environ, {"TEST_SECRET": str(valid)}, clear=True):
                self.assertEqual(secure_launch._read_secret_file("TEST_SECRET"), "a" * 48)
                self.assertNotIn("TEST_SECRET", os.environ)

            short = self._secret(directory, "short", "too-short")
            with mock.patch.dict(os.environ, {"TEST_SECRET": str(short)}, clear=True):
                with self.assertRaises(SystemExit):
                    secure_launch._read_secret_file("TEST_SECRET")

            link = directory / "link"
            link.symlink_to(valid)
            with mock.patch.dict(os.environ, {"TEST_SECRET": str(link)}, clear=True):
                with self.assertRaises(SystemExit):
                    secure_launch._read_secret_file("TEST_SECRET")

    def test_main_injects_secrets_in_memory_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            api = self._secret(directory, "api", "a" * 48)
            admin = self._secret(directory, "admin", "b" * 48)
            parsed = types.SimpleNamespace(
                grpc_mode=False,
                encoder_only=False,
                use_ray=False,
                enable_igw=False,
                tokenizer_worker_num=1,
            )
            ran = []
            ran_env = []
            prepared = []

            modules = {
                "sglang": types.ModuleType("sglang"),
                "sglang.srt": types.ModuleType("sglang.srt"),
                "sglang.launch_server": types.ModuleType("sglang.launch_server"),
                "sglang.srt.plugins": types.ModuleType("sglang.srt.plugins"),
                "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
                "sglang.srt.utils": types.ModuleType("sglang.srt.utils"),
            }
            def run_server(server_args):
                ran.append(server_args)
                ran_env.append(dict(os.environ))

            modules["sglang.launch_server"].run_server = run_server
            modules["sglang.srt.plugins"].load_plugins = lambda: None
            modules["sglang.srt.server_args"].prepare_server_args = (
                lambda args: prepared.append(args) or parsed
            )
            modules["sglang.srt.utils"].kill_process_tree = lambda *_, **__: None
            env = {
                "SGLANG_API_KEY_FILE": str(api),
                "SGLANG_ADMIN_API_KEY_FILE": str(admin),
                "SGLANG_TRUST_REMOTE_CODE": "0",
                "SGLANG_ENABLE_METRICS": "0",
                "SGLANG_MAX_QUEUED_REQUESTS": "32",
                "DUMPER_ENABLE": "1",
                "DUMPER_SERVER_PORT": "40000",
                "SGLANG_TEST_SCRIPTED_RUNTIME": "1",
                "SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR": "tcp://example:5555",
                "SGLANG_USE_PICKLE_IPC": "1",
            }
            argv = ["secure-launch.py", "--model-path", "/models/test"]
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(sys, "argv", argv),
            ):
                secure_launch.main()

            self.assertEqual(ran, [parsed])
            self.assertEqual(parsed.api_key, "a" * 48)
            self.assertEqual(parsed.admin_api_key, "b" * 48)
            self.assertFalse(parsed.trust_remote_code)
            self.assertFalse(parsed.enable_metrics)
            self.assertFalse(parsed.enable_custom_logit_processor)
            self.assertEqual(parsed.max_queued_requests, 32)
            self.assertIn("--max-queued-requests", prepared[0])
            self.assertNotIn("--trust-remote-code", prepared[0])
            self.assertNotIn("a" * 48, " ".join(argv))
            self.assertNotIn("b" * 48, " ".join(argv))
            self.assertNotIn("DUMPER_ENABLE", ran_env[0])
            self.assertNotIn("DUMPER_SERVER_PORT", ran_env[0])
            self.assertNotIn("SGLANG_TEST_SCRIPTED_RUNTIME", ran_env[0])
            self.assertNotIn("SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR", ran_env[0])
            self.assertEqual(ran_env[0]["SGLANG_USE_PICKLE_IPC"], "0")

    def test_abbreviated_config_option_is_rejected(self):
        with mock.patch.object(sys, "argv", ["secure-launch.py", "--conf=x.yaml"]):
            with self.assertRaises(SystemExit):
                secure_launch.main()

    def test_secondary_listener_modes_are_rejected(self):
        for argument in (
            "--language-only",
            "--enable-lora",
            "--disaggregation-mode=decode",
            "--tool-server=demo",
            "--nccl-init-addr=192.0.2.1:23456",
            "--kv-events-config={}",
            "--moe-a2a-backend=nixl",
            "--elastic-ep-backend=nixl",
        ):
            with self.subTest(argument=argument):
                with mock.patch.object(sys, "argv", ["secure-launch.py", argument]):
                    with self.assertRaises(SystemExit):
                        secure_launch.main()

    def test_parsed_remote_instance_loader_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            api = self._secret(directory, "api", "a" * 48)
            admin = self._secret(directory, "admin", "b" * 48)
            modules = {
                "sglang": types.ModuleType("sglang"),
                "sglang.srt": types.ModuleType("sglang.srt"),
                "sglang.launch_server": types.ModuleType("sglang.launch_server"),
                "sglang.srt.plugins": types.ModuleType("sglang.srt.plugins"),
                "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
                "sglang.srt.utils": types.ModuleType("sglang.srt.utils"),
            }
            modules["sglang.launch_server"].run_server = lambda _: self.fail(
                "unsafe parsed mode reached run_server"
            )
            modules["sglang.srt.plugins"].load_plugins = lambda: None
            modules["sglang.srt.server_args"].prepare_server_args = lambda _: parsed
            modules["sglang.srt.utils"].kill_process_tree = lambda *_, **__: None
            env = {
                "SGLANG_API_KEY_FILE": str(api),
                "SGLANG_ADMIN_API_KEY_FILE": str(admin),
            }
            for field, expected_name in (
                ("load_format", "remote_instance_loader"),
                (
                    "speculative_draft_load_format",
                    "speculative_remote_instance_loader",
                ),
            ):
                parsed = types.SimpleNamespace(**{field: "remote_instance"})
                with self.subTest(field=field):
                    with (
                        mock.patch.dict(sys.modules, modules),
                        mock.patch.dict(os.environ, env, clear=True),
                        mock.patch.object(
                            sys,
                            "argv",
                            ["secure-launch.py", "--model-path", "/models/test"],
                        ),
                    ):
                        with self.assertRaisesRegex(SystemExit, expected_name):
                            secure_launch.main()


if __name__ == "__main__":
    unittest.main()
