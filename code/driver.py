from exoplanet_knn import NearestNeighbor
from exoplanet_nn import NeuralNetwork
from exoplanet_tree import DecisionTree
import os

if __name__ == '__main__':
    knn = NearestNeighbor()
    nn = NeuralNetwork()
    tree = DecisionTree()

    print("loading data...")
    knn.load_data()
    nn.load_data()
    tree.load_data()

    ans = input("reset models? Y/N: ")
    if ans == "Y" or ans == "y":
        nn.delete_nn()
        knn.delete_knn()
        tree.delete_tree()

    if os.path.exists(nn.file):
        print("loading nn...")
        nn.load_nn()
    else:
        print("training nn...")
        nn.train()

    if os.path.exists(knn.file):
        print("loading knn...")
        knn.load_knn()
    else:
        print("creating knn model...")
        knn.train()

    if os.path.exists(tree.file):
        print("loading tree...")
        tree.load_tree()
    else:
        print("creating tree...")
        tree.train()

    print("predicting testing data on all methods...")
    knn.predict()
    nn.predict()
    tree.predict()

    print("\nAccuracy's: ")

    print("\nNeural Network accuracy: " + str(round(nn.accuracy, 2) * 100) + "%")
    print("Confusion Matrix: ")
    print(nn.matrix)

    print("\nK Nearest Neighbors accuracy: " + str(round(knn.accuracy, 2) * 100) + "%")
    print("Confusion Matrix: ")
    print(knn.matrix)

    print("\nDecision Tree accuracy: " + str(round(tree.accuracy, 2) * 100) + "%")
    print("Confusion Matrix: ")
    print(tree.matrix)

    ans = input("save files? Y/N: ")
    if ans == "Y" or ans == "y":
        print("saving models...")
        nn.save_nn()
        knn.save_knn()
        tree.save_tree()
