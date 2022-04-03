from arguments import Arguments


arguments = Arguments()
x = {
    "epochs": 1000,
}

arguments.load_from_dict(x)

print(arguments)