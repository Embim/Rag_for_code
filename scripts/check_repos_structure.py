"""Check repos directory structure."""
from pathlib import Path

repos_dir = Path("data/repos")

print(f"repos_dir exists: {repos_dir.exists()}")

if repos_dir.exists():
    contents = list(repos_dir.iterdir())
    print(f"Contents ({len(contents)} items):")
    for item in contents:
        print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")

    ui_bo = repos_dir / "ui"
    print(f"\nui exists: {ui_bo.exists()}")

    if ui_bo.exists():
        files = list(ui_bo.glob("**/*"))
        print(f"ui contents ({len(files)} items):")
        for f in files[:20]:
            print(f"  - {f.relative_to(repos_dir)}")

    api_bo = repos_dir / "api"
    print(f"\napi exists: {api_bo.exists()}")

    if api_bo.exists():
        files = list(api_bo.glob("**/*"))
        print(f"api contents ({len(files)} items):")
        for f in files[:20]:
            print(f"  - {f.relative_to(repos_dir)}")
else:
    print("repos_dir does not exist!")
