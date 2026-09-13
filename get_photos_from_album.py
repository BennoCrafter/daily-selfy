import subprocess
from pathlib import Path


def remove_old_photos(export_dir: Path) -> None:
    for item in export_dir.iterdir():
        item.unlink()


def get_photos_from_album(album_name: str, export_dir: Path) -> None:
    script = f'''
    tell application "Photos"
        set theAlbum to album "{album_name}"
        set theItems to media items of theAlbum
        export theItems to POSIX file "{export_dir.as_posix()}"
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    print("Return code:", result.returncode)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)


if __name__ == "__main__":
    import sys

    album_name = sys.argv[1]
    print(f"Getting photos from album: {album_name}")

    export_dir = Path.cwd() / "resources" / "images"
    export_dir.mkdir(parents=True, exist_ok=True)

    remove_old_photos(export_dir)
    get_photos_from_album(album_name, export_dir)
