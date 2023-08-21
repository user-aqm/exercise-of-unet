import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split()
            if len(values) == 2:
                data.append((float(values[0]), float(values[1])))
    return data

def plot_data(data):
    x_values = [item[1] for item in data]
    y_values = [item[0] for item in data]

    plt.plot(x_values, y_values, marker='o')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Data Plot")
    plt.show()
#
# file_path = "arrays.txt"  # 文件路径
# data = read_data_from_file(file_path)
# plot_data(data)


# 调用函数保存数组到文件
def save_arrays_to_file(filename, array1, array2):
    try:
        with open(filename, "w") as file:
            for a, b in zip(array1, array2):
                file.write(f"{a} {b}\n")
        print("Arrays saved successfully.")
    except Exception as e:
        print("Error while saving arrays:", e)


# save_arrays_to_file("arrays.txt", array1, array2)
