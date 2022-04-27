



token_count = 1023
path = "input_test"
token = "token"
with open(f"{path}_{token_count}.txt", "wt") as file:
    for i in range(1, token_count + 1):
        #if i % 10 == 0:
        #    file.write(f"{token}. ")
        #else:
        file.write(f"{token} ")
