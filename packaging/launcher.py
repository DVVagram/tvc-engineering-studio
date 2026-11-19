"""PyInstaller entry script (keeps the frozen app's imports explicit)."""

from rocket_sim.app import main

if __name__ == "__main__":
    raise SystemExit(main())
