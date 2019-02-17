import json

mode = "Unencrypted"

def prepare_data(x_train, x_test):
    return x_train, x_test


def print_results(test_acc, out, dataset, model, epochs, epoch_accuracies):
    out.write("{} {} {} {}\n".format(dataset, model, mode,test_acc))
    out.write(
        "\terror rate : {}%\n".format((1.0 - test_acc) * 100)
    )

    results = {
        "name": dataset+"_"+model+"_"+mode,
        "acc": epoch_accuracies,
        "epochs": epochs
    }

    # writing the training results
    with open("../{}_{}_models.json".format(mode,dataset), 'a') as j:
        json.dump(results, j)
        j.write('\n')
