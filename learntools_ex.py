from learntools.core import *
import numpy as np

# Q1: Create a one-dimensional NumPy array from 5 to 905 counting by 10.
q1_sol = np.array([  5,  15,  25,  35,  45,  55,  65,  75,  85,  95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255,
       265, 275, 285, 295, 305, 315, 325, 335, 345, 355, 365, 375, 385,
       395, 405, 415, 425, 435, 445, 455, 465, 475, 485, 495, 505, 515,
       525, 535, 545, 555, 565, 575, 585, 595, 605, 615, 625, 635, 645,
       655, 665, 675, 685, 695, 705, 715, 725, 735, 745, 755, 765, 775,
       785, 795, 805, 815, 825, 835, 845, 855, 865, 875, 885, 895])

# Q2: Create the same array using a Python range in a list comprehension.
q2_sol = np.array([  5,  15,  25,  35,  45,  55,  65,  75,  85,  95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255,
       265, 275, 285, 295, 305, 315, 325, 335, 345, 355, 365, 375, 385,
       395, 405, 415, 425, 435, 445, 455, 465, 475, 485, 495, 505, 515,
       525, 535, 545, 555, 565, 575, 585, 595, 605, 615, 625, 635, 645,
       655, 665, 675, 685, 695, 705, 715, 725, 735, 745, 755, 765, 775,
       785, 795, 805, 815, 825, 835, 845, 855, 865, 875, 885, 895])

# Q3: Create a NumPy array of the capital letters A-Z.
q3_sol = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
# Alternatively: np.array(list(string.ascii_uppercase))

# Q4: Create a ten-element array of zeros and a six-element array of ones.
# Here we combine the two arrays into a tuple.
q4_sol_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
q4_sol_2 = np.array([1, 1, 1, 1, 1, 1])

# Q5: Given a 2D array, extract the third column.
input_q5 = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
q5_sol = np.array([33, 66, 99])

# Q6: Create an array of shape (3, 4) filled with 64-bit integer zeros.
q6_sol = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64)

# Q7: Split an array (reshaped to 9x3) into 3 equal sub-arrays.
arr_q7 = np.arange(7, 34, 1).reshape(9, 3)
q7_sol_1 = np.array([[ 7,  8,  9], [10, 11, 12], [13, 14, 15]])
q7_sol_2 = np.array([[16, 17, 18], [19, 20, 21], [22, 23, 24]])
q7_sol_3 = np.array([[25, 26, 27], [28, 29, 30], [31, 32, 33]])

# Q8: Return the last two elements of each of the first rows (all but the last row) in a (3,4) array.
input_q8 = np.arange(1, 13, 1).reshape(3, 4)
q8_sol = np.array([[3, 4], [7, 8]])

# Q9: Count the number of elements greater than 5.
input_q9 = np.arange(1, 13, 1).reshape(3, 4)
q9_sol = 7

# Q10: Delete the second column from an array and insert a new column in its place.
input_q10 = np.array([[34, 72, 33], [82, 22, 32], [59, 84, 10]])
insert_data = np.array([[37, 37, 37]])
temp = np.delete(input_q10, 1, axis=1)
q10_sol = np.array([[34, 37, 33], [82, 37, 32], [59, 37, 10]])

# Q11: Compute Euclidean distances between consecutive points and append as a new column.
input_q11 = np.array([[1, 2], [4, 6], [7, 8], [10, 10]])
diffs = np.diff(input_q11, axis=0)
distance = np.sqrt(np.sum(diffs**2, axis=1))
distance = np.append(distance, np.nan)  # Pad last row with 0
q11_sol = np.array([[ 1.,  2.,  5.], [ 4.,  6.,  3.60555128], [ 7.,  8.,  3.60555128], [10., 10., 0]])

# Q12: Remove consecutive duplicate rows from an array.
arr_q12 = np.array([[1, 1], [2, 1], [3, 3], [3, 3], [2, 1], [1, 1]])
mask = np.insert(np.any(arr_q12[1:] != arr_q12[:-1], axis=1), 0, True)
q12_sol = np.array([[1, 1], [2, 1], [3, 3], [2, 1], [1, 1]])

# Q13: Normalize a 2D array by subtracting the mean and dividing by the std (avoid division by zero).
strokes = np.array([[10, 20, 30], [15, 24, 33], [8, 18, 29], [14, 22, 32]])
q13_sol = np.array([[-0.61159284, -0.4472136 , -0.63245553],
                    [ 1.13581527,  1.34164079,  1.26491106],
                    [-1.31055608, -1.34164079, -1.26491106],
                    [ 0.78633365,  0.4472136 ,  0.63245553]])

# Q14: Delete the second column from an array and insert a new column where each value is the row sum.
arr_q14 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_sums = np.sum(arr_q14, axis=1)
temp_q14 = np.delete(arr_q14, 1, axis=1)
q14_sol = np.array([[ 1,  6,  3],
                  [ 4, 15,  6],
                  [ 7, 24,  9]])

# Q15: Extract unique characters from an array of strings.
arr_q15 = np.array(["hello", "world", "numpy", "rocks"])
unique_chars = set(sorted("".join(arr_q15)))
q15_sol = set(['c', 'd', 'e', 'h', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 'u', 'w', 'y'])

# Q16: Generate a dictionary mapping each unique character to a unique index (sorted by ASCII).
chars = np.array(["a", "b", "c", "d", "a", "c", "e"])
q16_sol = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
# Expected: {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

# Q17: Stack a list of 2D traces into a single NumPy array.
traces = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8], [9, 10]])]
q17_sol = np.array([[ 1,  2], [ 3,  4], [ 5,  6], [ 7,  8], [ 9, 10]])

# Q18: Convert a list of text labels into a NumPy array of integer encodings.
vocab = {
    "hello": 0,
    "numpy": 1,
    "rocks": 2,
    "world": 3,
    "python": 4,
    "data": 5,
    "science": 6,
}
labels = "hello rocks"
q18_sol = np.array([0, 2])

# Q19: Extract non-zero differences from a given array.
delta = np.array([[3, 4], [0, 0], [-2, 1]])
mask_q19 = np.all(delta == 0, axis=1)
q19_sol = np.array([[ 3,  4], [-2,  1]])

# Q20: Time Series Data Transformation and Feature Engineering.
data = np.array(
    [
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 10000.0],  # Outlier in third column
        [4.0, 40.0, 400.0],
        [5.0, 50.0, 500.0],
    ]
)
# 1. Normalize the data.
mean_data = np.mean(data, axis=0)
std_data = np.std(data, axis=0)
std_data[std_data == 0] = 1
normalized_data = (data - mean_data) / std_data
# 2. Detect anomalies (>2.5 std) and mask them with NaN.
anomalies = np.abs(normalized_data) > 2.5
data_with_nan = data.copy()
data_with_nan[anomalies] = np.nan
# 3. Interpolate missing values (fill NaN with column mean).
interpolated_data = np.where(
    np.isnan(data_with_nan), np.nanmean(data_with_nan, axis=0), data_with_nan
)
# 4. Compute first-order differences.
first_order_diff = np.diff(interpolated_data, axis=0)
first_order_diff_padded = np.vstack(
    (first_order_diff, np.zeros((1, first_order_diff.shape[1])))
)
# 5. Create a missing flag.
missing_flag = np.isnan(data_with_nan).astype(int)
# 6. Stack features into a final matrix.
q20_sol = np.array([[ 1.0e+00,  1.0e+01,  1.0e+02,  1.0e+00,  1.0e+01,  1.0e+02,
         0.0e+00,  0.0e+00,  0.0e+00],
       [ 2.0e+00,  2.0e+01,  2.0e+02,  1.0e+00,  1.0e+01,  9.8e+03,
         0.0e+00,  0.0e+00,  0.0e+00],
       [ 3.0e+00,  3.0e+01,  1.0e+04,  1.0e+00,  1.0e+01, -9.6e+03,
         0.0e+00,  0.0e+00,  0.0e+00],
       [ 4.0e+00,  4.0e+01,  4.0e+02,  1.0e+00,  1.0e+01,  1.0e+02,
         0.0e+00,  0.0e+00,  0.0e+00],
       [ 5.0e+00,  5.0e+01,  5.0e+02,  0.0e+00,  0.0e+00,  0.0e+00,
         0.0e+00,  0.0e+00,  0.0e+00]])
# Expected: a (5, 9) matrix containing the interpolated data, first-order differences, and missing flags.


# --- Exercise Classes ---
class Exercise1(EqualityCheckProblem):
    _solution = None
    _var = "q1_sol"
    _expected = q1_sol
    _hint = "Use np.arange(start, stop, step) to create the array."


class Exercise2(EqualityCheckProblem):
    _solution = None
    _var = "q2_sol"
    _expected = q2_sol
    _hint = "Use a list comprehension with range and convert it to a numpy array."


class Exercise3(EqualityCheckProblem):
    _solution = None
    _var = "q3_sol"
    _expected = q3_sol
    _hint = "Consider using string.ascii_uppercase or chr() with ord() in a list comprehension."


class Exercise4(EqualityCheckProblem):
    _solution = None
    _vars = ["q4_sol_1", "q4_sol_2"]
    _expected = [q4_sol_1, q4_sol_2]
    _hint = "Use np.zeros and np.ones to create the arrays and return them as a tuple."


class Exercise5(EqualityCheckProblem):
    _solution = None
    _var = "q5_sol"
    _expected = q5_sol
    _hint = "Index the third column (column index 2) using slicing."


class Exercise6(EqualityCheckProblem):
    _solution = None
    _var = "q6_sol"
    _expected = q6_sol
    _hint = "Use np.zeros with the correct dtype and shape."


class Exercise7(EqualityCheckProblem):
    _solution = None
    _vars = ["q7_sol_1", "q7_sol_2", "q7_sol_3"]
    _expected = [q7_sol_1, q7_sol_2, q7_sol_3]
    _hint = "Reshape the array and use np.split() to divide it into three parts."


class Exercise8(EqualityCheckProblem):
    _solution = None
    _var = "q8_sol"
    _expected = q8_sol
    _hint = "Slice the array to select the first rows and then the last two columns."


class Exercise9(EqualityCheckProblem):
    _solution = None
    _var = "q9_sol"
    _expected = q9_sol
    _hint = "Apply a boolean mask to count elements greater than 5."


class Exercise10(EqualityCheckProblem):
    _solution = None
    _var = "q10_sol"
    _expected = q10_sol
    _hint = "Use np.delete() to remove a column and np.insert() to add a new column."


class Exercise11(EqualityCheckProblem):
    _solution = None
    _var = "q11_sol"
    _expected = q11_sol
    _hint = "Compute differences with np.diff(), then calculate Euclidean distances and append 0."


class Exercise12(EqualityCheckProblem):
    _solution = None
    _var = "q12_sol"
    _expected = q12_sol
    _hint = "Construct a boolean mask to filter out consecutive duplicate rows."


class Exercise13(EqualityCheckProblem):
    _solution = None
    _var = "q13_sol"
    _expected = q13_sol
    _hint = "Subtract the column means and divide by the standard deviations (ensure no division by zero)."


class Exercise14(EqualityCheckProblem):
    _solution = None
    _var = "q14_sol"
    _expected = q14_sol
    _hint = "Compute the row sums, remove the second column, and insert the sums at that index."


class Exercise15(EqualityCheckProblem):
    _solution = None
    _var = "q15_sol"
    _expected = q15_sol
    _hint = "Concatenate the strings, build a set of unique characters, and sort them."


class Exercise16(EqualityCheckProblem):
    _solution = None
    _var = "q16_sol"
    _expected = q16_sol
    _hint = "Use sorted(set(chars)) with enumerate to construct the dictionary mapping."


class Exercise17(EqualityCheckProblem):
    _solution = None
    _var = "q17_sol"
    _expected = q17_sol
    _hint = "Stack the list of arrays using np.vstack()."


class Exercise18(EqualityCheckProblem):
    _solution = None
    _var = "q18_sol"
    _expected = q18_sol
    _hint = "Split the labels string and map each label to its corresponding index via the vocabulary dictionary."


class Exercise19(EqualityCheckProblem):
    _solution = None
    _var = "q19_sol"
    _expected = q19_sol
    _hint = "Create a mask to filter out rows that are all zeros and use it to index the array."


class Exercise20(EqualityCheckProblem):
    _solution = None
    _var = "q20_sol"
    _expected = q20_sol
    _hint = (
        "Perform normalization, detect anomalies, interpolate missing values, compute first-order differences, "
        "and finally stack the features along with a missing data flag."
    )


qvars = bind_exercises(
    globals(),
    [
        Exercise1,
        Exercise2,
        Exercise3,
        Exercise4,
        Exercise5,
        Exercise6,
        Exercise7,
        Exercise8,
        Exercise9,
        Exercise10,
        Exercise11,
        Exercise12,
        Exercise13,
        Exercise14,
        Exercise15,
        Exercise16,
        Exercise17,
        Exercise18,
        Exercise19,
        Exercise20,
    ],
    var_format="q{n}",
)
__all__ = list(qvars)
