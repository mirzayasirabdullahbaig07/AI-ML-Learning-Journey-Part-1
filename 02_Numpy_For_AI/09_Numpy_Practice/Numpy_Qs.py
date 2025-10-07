import numpy as np

# 1. What is NumPy and why is it important for data science and machine learning?
# Definition: NumPy is a library for fast numerical computation using multidimensional arrays.
# Real-world Example: Used in ML models for matrix operations and data preprocessing.
arr = np.array([1, 2, 3])
print(arr)

# 2. How do you create a NumPy array from a Python list?
# Definition: Use np.array() to convert a list into a NumPy array.
# Real-world Example: Converting raw data lists into arrays for ML models.
arr = np.array([10, 20, 30])
print(arr)

# 3. What is the difference between a Python list and a NumPy array?
# Definition: Lists store heterogeneous data; NumPy arrays store homogeneous data and are faster.
# Real-world Example: Large numerical datasets handled faster with NumPy arrays.
list_ = [1, 2, 3]
array_ = np.array([1, 2, 3])
print(type(list_), type(array_))

# 4. Explain the concept of vectorization in NumPy.
# Definition: Performing operations on entire arrays without explicit loops.
# Real-world Example: Vectorized operations speed up ML computations.
arr = np.array([1, 2, 3, 4])
print(arr * 2)

# 5. What is broadcasting in NumPy and how does it work?
# Definition: Allows operations between arrays of different shapes by expanding dimensions.
# Real-world Example: Adding a constant to every element of an array.
arr = np.array([1, 2, 3])
print(arr + 5)

# 6. How do you generate random numbers using NumPy?
# Definition: Use np.random methods to generate random values.
# Real-world Example: Used for initializing weights in neural networks.
rand_nums = np.random.rand(3)
print(rand_nums)

# 7. What is the difference between np.arange() and np.linspace()?
# Definition: arange() uses step size, linspace() uses number of samples.
# Real-world Example: Creating training steps or simulation ranges.
print(np.arange(0, 10, 2))
print(np.linspace(0, 10, 5))

# 8. How do you find the shape and number of dimensions of a NumPy array?
# Definition: shape gives dimensions; ndim gives number of dimensions.
# Real-world Example: Understanding data structure before feeding into ML model.
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape, arr.ndim)

# 9. What is the difference between reshape() and resize() in NumPy?
# Definition: reshape() returns new array; resize() modifies in-place.
# Real-world Example: Preparing input shapes for ML algorithms.
a = np.arange(6)
print(a.reshape(2, 3))
a.resize(3, 2)
print(a)

# 10. How can you combine two arrays vertically and horizontally?
# Definition: Use vstack() for vertical, hstack() for horizontal stacking.
# Real-world Example: Merging datasets or feature arrays.
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.vstack((a, b)))
print(np.hstack((a, b)))

# 11. What are universal functions (ufuncs) in NumPy? Give examples.
# Definition: Functions that operate element-wise on arrays.
# Real-world Example: Applying math operations on entire datasets.
arr = np.array([1, 4, 9, 16])
print(np.sqrt(arr))
print(np.exp(arr))

# 12. How can you perform element-wise arithmetic operations on arrays?
# Definition: Use +, -, *, / for element-wise operations.
# Real-world Example: Feature scaling or mathematical transformations.
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)
print(a * b)

# 13. What is the difference between np.sum() and np.cumsum()?
# Definition: sum() gives total; cumsum() gives cumulative sum.
# Real-world Example: Calculating running totals.
arr = np.array([1, 2, 3, 4])
print(np.sum(arr))
print(np.cumsum(arr))

# 14. How do you find the mean, median, and standard deviation of an array?
# Definition: Use np.mean(), np.median(), np.std().
# Real-world Example: Descriptive statistics for dataset.
arr = np.array([1, 2, 3, 4, 5])
print(np.mean(arr))
print(np.median(arr))
print(np.std(arr))

# 15. How do you find the index of the maximum and minimum values in an array?
# Definition: Use np.argmax() and np.argmin().
# Real-world Example: Locating highest and lowest feature values.
arr = np.array([10, 20, 5, 25, 15])
print(np.argmax(arr))
print(np.argmin(arr))

# 16. What does the np.where() function do?
# Definition: Returns indices where a condition is True.
# Real-world Example: Filtering data points based on conditions.
arr = np.array([10, 15, 20, 25, 30])
print(np.where(arr > 20))

# 17. How can you filter elements of an array based on a condition?
# Definition: Use boolean indexing.
# Real-world Example: Selecting data above a threshold.
arr = np.array([10, 20, 30, 40, 50])
filtered = arr[arr > 25]
print(filtered)

# 18. How can you flatten a 2D array into a 1D array?
# Definition: Use flatten() or ravel().
# Real-world Example: Preparing image data for ML models.
arr = np.array([[1, 2], [3, 4]])
print(arr.flatten())

# 19. What is the difference between shallow copy and deep copy in NumPy?
# Definition: Shallow copy shares data; deep copy creates independent copy.
# Real-world Example: Avoid accidental data modification in ML pipelines.
a = np.array([1, 2, 3])
b = a.view()  # shallow copy
c = a.copy()  # deep copy
a[0] = 100
print(b)
print(c)

# 20. How do you stack arrays using vstack() and hstack()?
# Definition: vstack() stacks vertically, hstack() horizontally.
# Real-world Example: Combining multiple features.
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.vstack((a, b)))
print(np.hstack((a, b)))


import numpy as np

# 21. How do you compute the dot product of two arrays?
# Definition: Use np.dot() to calculate the sum of element-wise products.
# Real-world Example: Used in linear regression and neural networks.
a = np.array([1, 2])
b = np.array([3, 4])
print(np.dot(a, b))

# 22. What is the difference between np.dot() and np.matmul()?
# Definition: np.dot() works for 1D/2D arrays; np.matmul() strictly for matrices.
# Real-world Example: Used in deep learning matrix multiplications.
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
print(np.dot(A, B))
print(np.matmul(A, B))

# 23. How can you compute the determinant of a matrix using NumPy?
# Definition: Use np.linalg.det() for determinant.
# Real-world Example: Used in checking matrix invertibility.
A = np.array([[1, 2], [3, 4]])
print(np.linalg.det(A))

# 24. How do you calculate the inverse of a matrix using NumPy?
# Definition: Use np.linalg.inv() for matrix inverse.
# Real-world Example: Used in solving linear equations.
A = np.array([[1, 2], [3, 4]])
print(np.linalg.inv(A))

# 25. How do you find the eigenvalues and eigenvectors of a matrix?
# Definition: Use np.linalg.eig() to compute both.
# Real-world Example: Used in PCA for dimensionality reduction.
A = np.array([[2, 0], [0, 3]])
vals, vecs = np.linalg.eig(A)
print(vals)
print(vecs)

# 26. How can you generate an identity matrix in NumPy?
# Definition: Use np.eye(n) to create an n x n identity matrix.
# Real-world Example: Used in linear algebra and ML model updates.
print(np.eye(3))

# 27. What is the purpose of the np.diag() function?
# Definition: Extracts or creates a diagonal matrix.
# Real-world Example: Extracting main diagonal from covariance matrix.
A = np.array([[1, 2], [3, 4]])
print(np.diag(A))

# 28. How do you generate an array filled with zeros or ones?
# Definition: Use np.zeros() or np.ones().
# Real-world Example: Initializing bias and weights in neural networks.
print(np.zeros((2, 3)))
print(np.ones((2, 3)))

# 29. What is the difference between np.empty() and np.zeros()?
# Definition: np.empty() creates uninitialized array; np.zeros() fills with zeros.
# Real-world Example: np.empty() used when speed is preferred over initialization.
print(np.empty((2, 2)))
print(np.zeros((2, 2)))

# 30. How do you use np.random.seed() and why is it important?
# Definition: Sets seed for reproducible random results.
# Real-world Example: Ensures consistent results in experiments.
np.random.seed(42)
print(np.random.rand(3))

# 31. What is a structured array in NumPy?
# Definition: Arrays with columns of different data types.
# Real-world Example: Used to store tabular data.
data = np.array([(1, 'Alice', 25.0), (2, 'Bob', 30.5)], dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'f4')])
print(data)

# 32. How can you sort an array and get the sorted indices?
# Definition: Use np.sort() to sort, np.argsort() to get sorted indices.
# Real-world Example: Sorting feature importance scores.
arr = np.array([40, 10, 30, 20])
print(np.sort(arr))
print(np.argsort(arr))

# 33. What does the np.unique() function return?
# Definition: Returns unique elements and optionally their counts.
# Real-world Example: Finding unique categories in data.
arr = np.array([1, 2, 2, 3, 3, 3])
print(np.unique(arr, return_counts=True))

# 34. How can you find the intersection and union of two arrays?
# Definition: Use np.intersect1d() and np.union1d().
# Real-world Example: Comparing datasets or feature sets.
a = np.array([1, 2, 3])
b = np.array([3, 4, 5])
print(np.intersect1d(a, b))
print(np.union1d(a, b))

# 35. What is broadcasting error and how can it be resolved?
# Definition: Occurs when array shapes don’t align for operation.
# Real-world Example: Fix by reshaping or matching dimensions.
a = np.array([1, 2, 3])
# b = np.array([[1, 2], [3, 4]])  # incompatible
# Solution: reshape a
# print(a.reshape(3, 1) + b)

# 36. How do you read data from a CSV file using NumPy?
# Definition: Use np.genfromtxt() or np.loadtxt().
# Real-world Example: Loading numerical datasets.
# data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

# 37. How do you save and load NumPy arrays to a binary file?
# Definition: Use np.save() and np.load().
# Real-world Example: Saving preprocessed data for ML reuse.
arr = np.array([1, 2, 3])
np.save('myarray.npy', arr)
print(np.load('myarray.npy'))

# 38. What is the purpose of the np.nditer() function?
# Definition: Efficient iterator for array traversal.
# Real-world Example: Looping efficiently over large datasets.
arr = np.array([[1, 2], [3, 4]])
for x in np.nditer(arr):
    print(x, end=' ')
print()

# 39. How can you normalize an array in NumPy?
# Definition: Scale array values between 0 and 1.
# Real-world Example: Feature scaling before ML training.
arr = np.array([10, 20, 30, 40])
norm = (arr - arr.min()) / (arr.max() - arr.min())
print(norm)

# 40. What is the difference between np.any() and np.all()?
# Definition: any() checks if any True; all() checks if all True.
# Real-world Example: Checking conditions in datasets.
arr = np.array([True, False, True])
print(np.any(arr))
print(np.all(arr))

# 41. How do you change the data type of an array?
# Definition: Use astype() to cast to another dtype.
# Real-world Example: Converting data types for model compatibility.
arr = np.array([1.1, 2.2, 3.3])
print(arr.astype(int))

# 42. How can you check if two NumPy arrays are equal?
# Definition: Use np.array_equal().
# Real-world Example: Comparing model predictions.
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
print(np.array_equal(a, b))

# 43. What is memory layout (C-contiguous vs F-contiguous) in NumPy arrays?
# Definition: C-order stores row-wise, F-order column-wise.
# Real-world Example: Affects performance in ML operations.
arr_c = np.array([[1, 2], [3, 4]], order='C')
arr_f = np.array([[1, 2], [3, 4]], order='F')
print(arr_c.flags['C_CONTIGUOUS'], arr_f.flags['F_CONTIGUOUS'])

# 44. What is the difference between axis=0 and axis=1?
# Definition: axis=0 means columns; axis=1 means rows.
# Real-world Example: Summing features or samples.
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(arr, axis=0))
print(np.sum(arr, axis=1))

# 45. How do you handle missing or NaN values in a NumPy array?
# Definition: Use np.isnan(), np.nan_to_num(), or remove NaNs.
# Real-world Example: Cleaning data before ML processing.
arr = np.array([1, np.nan, 3, np.nan])
print(np.isnan(arr))
print(np.nan_to_num(arr))

# 46. How can you generate a diagonal matrix in NumPy?
# Definition: Use np.diag() with a 1D array.
# Real-world Example: Representing feature weights.
arr = np.array([1, 2, 3])
print(np.diag(arr))

# 47. What does the np.clip() function do?
# Definition: Limits array values between min and max.
# Real-world Example: Preventing overflow or outlier effects.
arr = np.array([1, 5, 10, 15])
print(np.clip(arr, 5, 10))

# 48. How can you repeat or tile arrays in NumPy?
# Definition: Use np.repeat() or np.tile().
# Real-world Example: Data augmentation or matrix replication.
arr = np.array([1, 2, 3])
print(np.repeat(arr, 2))
print(np.tile(arr, 2))

# 49. How do you find percentiles or quantiles in a NumPy array?
# Definition: Use np.percentile() or np.quantile().
# Real-world Example: Analyzing distribution in datasets.
arr = np.array([1, 2, 3, 4, 5])
print(np.percentile(arr, 50))
print(np.quantile(arr, 0.25))

# 50. How do you perform matrix multiplication on two arrays using NumPy?
# Definition: Use @ operator or np.matmul().
# Real-world Example: Core of linear regression and neural networks.
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)
print(np.matmul(A, B))
