import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import time

def solve_linear_system(A, b):
    """
    Решает систему линейных уравнений Ax = b
    где A - квадратная матрица, b - вектор значений.
    """
    try:
        x = np.linalg.solve(A, b)
        return x
    except np.linalg.LinAlgError as e:
        return None

def parallel_solve(A, b, n_experiments, n_workers=4):
    start_time_parallel_creation = time.time()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(solve_linear_system, [A] * n_experiments, [b] * n_experiments))
    end_time_parallel_creation = time.time()
    creation_time = end_time_parallel_creation - start_time_parallel_creation
    st.write(f"Время на создание и выполнение процессов: {creation_time:.6f} секунд")
    return results



st.title("Решение СЛАУ")
st.write("В этом эксперименте можно выбрать количество решений СЛАУ и размерность матрицы для демонстрации скорости параллельного и последовательного подходов.")
#st.sidebar.empty() 

rows = st.slider("Размер матрицы A (n x n)", 2, 1000, value=3)
A = np.random.rand(rows, rows)
b = np.random.rand(rows)

n_experiments = st.number_input("Количество повторений решения СЛАУ", min_value=1, max_value=100, value=50, step=1)

if st.button("Решить СЛАУ"):
    st.write("Матрица A:", A)
    st.write("Вектор b:", b)
    st.write(f"Выполнение {n_experiments} повторений")

    start_time_seq = time.time()
    for _ in range(n_experiments):
        solve_linear_system(A, b)
    end_time_seq = time.time()
    avg_seq_duration = (end_time_seq - start_time_seq) / n_experiments
    total_seq_duration = end_time_seq - start_time_seq

    start_time_parallel = time.time()
    parallel_solve(A, b, n_experiments)
    end_time_parallel = time.time()
    avg_par_duration = (end_time_parallel - start_time_parallel) / n_experiments
    total_par_duration = end_time_parallel - start_time_parallel

    st.write(f"Среднее время выполнения (последовательно): {avg_seq_duration:.6f} секунд на одно решение")
    st.write(f"Среднее время выполнения (параллельно): {avg_par_duration:.6f} секунд на одно решение")

    st.write(f"Общее время выполнения (последовательно): {total_seq_duration:.6f} секунд")
    st.write(f"Общее время выполнения (параллельно): {total_par_duration:.6f} секунд")

    fig, ax = plt.subplots()
    ax.bar(["Последовательное", "Параллельное"], [avg_seq_duration, avg_par_duration], color=['blue', 'green'])
    ax.set_ylabel("Среднее время (секунды)")
    ax.set_title("Сравнение среднего времени выполнения")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.bar(["Последовательное", "Параллельное"], [total_seq_duration, total_par_duration], color=['blue', 'green'])
    ax2.set_ylabel("Общее время (секунды)")
    ax2.set_title("Сравнение общего времени выполнения")
    st.pyplot(fig2)
