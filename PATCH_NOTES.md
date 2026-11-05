# Compatibility changes and source attribution

The `OptIForest-main/` subdirectory is a copy of the upstream implementation from https://github.com/xiagll/OptIForest and is not original student-written code. The proposal identifies upstream commit `a8be90d3ba767ead91bfc054cdea562327c11a31`, but the supplied ZIP did not carry a `.git` directory allowing its commit to be verified.

For NumPy >= 2 compatibility, use `np.asmatrix(` in place of removed `np.mat(`. The supplied upstream `opt_tree.py` already had one such compatibility modification. In this packaging step, the one remaining `np.mat(` call in `detectors/opt_forest.py` was replaced with `np.asmatrix(`. This packaging change was not followed by a full OptIForest rerun and should be independently checked. The change is narrow but this does not prove behavioural identity across all paths.

The runtime warning in `detectors/opt_tree.py` during tree construction has not been resolved. Original upstream code and license should be reviewed before redistribution.
