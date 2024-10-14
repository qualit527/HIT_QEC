import numpy as np

def show_stabilizers(Hx, Hz):
    countx = 0
    countz = 0
    print("Hx stabilizers:")
    for r in range(Hx.shape[0]):
        nonZeroElementsHx = np.where(Hx[r, :] != 0)[0] + 1
        countx += len(nonZeroElementsHx)
        print(f"Row {r + 1}: {nonZeroElementsHx}")
    print("Hz stabilizers:")
    for r in range(Hz.shape[0]):
        nonZeroElementsHz = np.where(Hz[r, :] != 0)[0] + 1
        countz += len(nonZeroElementsHz)
        print(f"Row {r + 1}: {nonZeroElementsHz}")

    print("countx: ", countx)
    print("countz: ", countz)