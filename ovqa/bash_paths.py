"""
Prints all environment variables in a format that can be used in a bash script.

Example:
    source <(python -m ovqa.bash_paths)
"""

from ovqa.paths import print_all_environment_variables


def main():
    print_all_environment_variables(prefix="export ")


if __name__ == "__main__":
    main()
