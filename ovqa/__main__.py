from packg.packaging import run_package


def main():
    run_package(__file__, "private/run")


if __name__ == "__main__":
    main()
