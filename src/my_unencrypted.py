import json

mode = "MY_UNENCRYPTED"


def prepare_data(inputs):
    return inputs


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
