import json
import os
import subprocess

import freetensor as ft


def make_simplifiable_ast():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 0 * i
    return ft.pop_ast()


def make_split_ast():
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8, label="L1") as i:
            y[i] = i
    return ft.pop_ast()


def run_transform(args, input_ast=None, **kwargs):
    input_text = None if input_ast is None else ft.dump_ast(input_ast)
    return subprocess.run(
        ["freetensor-transform", *args],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        **kwargs,
    )


def test_help():
    result = run_transform(["--help"])
    assert "Usage:" in result.stdout
    assert "freetensor-transform" in result.stdout


def test_pass_stdin_stdout():
    ast = make_simplifiable_ast()
    result = run_transform(["simplify"], ast)
    ast_text, info_text = result.stdout.rsplit("\n", 2)[:2]

    out = ft.load_ast(ast_text + "\n")
    expected = ft.simplify(ast)
    assert out.match(expected)
    assert json.loads(info_text) == {}


def test_file_input_output_and_info_file(tmp_path):
    ast = make_simplifiable_ast()
    input_file = tmp_path / "input.ast"
    output_file = tmp_path / "output.ast"
    info_file = tmp_path / "info.json"
    input_file.write_text(ft.dump_ast(ast))

    run_transform([
        "pb-simplify",
        "-i",
        str(input_file),
        "-o",
        str(output_file),
        "--info-file",
        str(info_file),
    ])

    out = ft.load_ast(output_file.read_text())
    expected = ft.pb_simplify(ast)
    assert out.match(expected)
    assert json.loads(info_file.read_text()) == {}


def test_info_fd_and_kebab_case_alias(tmp_path):
    ast = make_split_ast()
    loop = int(ft.Schedule(ast).find("L1").id)
    input_file = tmp_path / "input.ast"
    output_file = tmp_path / "output.ast"
    info_file = tmp_path / "info.json"
    input_file.write_text(ft.dump_ast(ast))

    fd = os.open(info_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        run_transform([
            "split",
            "-i",
            str(input_file),
            "-o",
            str(output_file),
            "--info-fd",
            str(fd),
            "--id",
            str(loop),
            "--factor=4",
        ],
                      pass_fds=(fd,))
    finally:
        os.close(fd)

    envelope = json.loads(info_file.read_text())
    assert envelope["ok"] is True
    info = envelope["info"]
    assert set(info) == {"outer", "inner"}
    assert info["outer"]
    assert info["inner"]

    out = ft.load_ast(output_file.read_text())
    expected = ft.Schedule(ast)
    expected.split("L1", 4)
    assert out.match(expected.ast())
