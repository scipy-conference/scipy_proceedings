from pathlib import Path
import shutil
import sys, subprocess, os
import argparse

def _print_and_run(*cmd, **kwargs):
    print(*cmd)
    return subprocess.run(cmd, **kwargs)


def _find_root_path():
    cur_rootpath = paper_dir
    while not cur_rootpath.joinpath(".git").exists():
        cur_rootpath = cur_rootpath.parent

    if not cur_rootpath.joinpath("publisher").exists():
        print(f"invalid root found: {cur_rootpath}")
        sys.exit(1)
    else:
        print(f"Found rootpath: {cur_rootpath}")
    return cur_rootpath


def _ensure_overleaf_remote_exists():
    existing_remotes = subprocess.run(
        ["git", "remote"], capture_output=True, text=True, cwd=rootpath
    ).stdout.split()
    if "overleaf" in existing_remotes:
        # Nothing more to do
        return
    else:
        repo_name = os.environ.get("OVERLEAF_CURRENT_REPO", None)
        if not repo_name:
            raise ValueError("Must fetch from overleaf, but remote was not set!")
        print("Overleaf remote didn't already exist, adding now")
        _print_and_run(
            f"git remote add overleaf {repo_name}"
        )


def rm_old_outputs():
    if (outpath := rootpath / "output" / paper_id).exists():
        print("Removing existing output directory")
        shutil.rmtree(outpath)


def sync_overleaf():
    _ensure_overleaf_remote_exists()

    output = _print_and_run("git", "fetch", "overleaf", cwd=rootpath, check=True, capture_output=True, text=True)
    # if output.stdout == "":
        # print("Overleaf remote is up to date, no need to fetch figures and sections")
        # return

    worktree_dir = os.path.relpath(paper_dir, rootpath)
    for pull_dir in "figures", "sections":
        _print_and_run(
            "git",
            f"--work-tree={worktree_dir}",
            "checkout",
            f"overleaf/master",
            "--",
            pull_dir,
            cwd=rootpath,
            check=True,
        )


def build_paper():
    builder_script = rootpath / "publisher/build_paper.py"
    paper_relpath = f"papers/{paper_id}"
    subprocess.run(
        [sys.executable, builder_script, paper_relpath], shell=True, cwd=rootpath
    )


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync-overleaf",
        action="store_true",
        default=False,
        help="Check for overleaf upstream and fetch changes"
    )
    return parser

paper_id = "221_jessurun"
paper_dir = Path(__file__).resolve().parent
rootpath = _find_root_path()

if __name__ == "__main__":
    parser = create_argparser()
    if parser.parse_args().sync_overleaf:
        sync_overleaf()
    rm_old_outputs()
    build_paper()
