from Algorithm import Algorithm

maxBlocks = 5


def main():
    algorithm = Algorithm(maxBlocks, None, None, None, None, max_attempts=5)

    cifar_10 = algorithm.generate_model(is_cifar=True)


if __name__ == "__main__":
    main()
