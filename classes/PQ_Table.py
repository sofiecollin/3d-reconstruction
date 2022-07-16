import numpy as np

# Uses a 2D list. 

class PQ_Table:
    def __init__(self):
        self.table = []
        self.V = []

    def add_row(self, points_2d):
        self.table.append(list())
        self.V.append(list())
        for p in points_2d.T:
            self.table[-1].append(p)
            
            if p[0] == -1 and p[1] == -1:
                self.V[-1].append(False)
            else:
                self.V[-1].append(True)

    def get(self, row, col):
        if row >= len(self.table) or col >= len(self.table[0]):
            raise ValueError('Trying to index outside of table. Size of table is: ' + str(len(self.table)) + ' x ' + str(len(self.table[0])))

        return self.table[row][col]

    def get_mtx(self):

        table_mtx = np.zeros((len(self.table), len(self.table[0]), 2))

        # corr1
        table_mtx[0, :] = np.asarray(self.table[0])
        # corr2
        table_mtx[1, :] = np.asarray(self.table[1])

        return table_mtx