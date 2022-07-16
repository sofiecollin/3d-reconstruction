import numpy as np

class Q:
    def __init__(self):
        self.views = []

    def add_view(self, view):
        if view.R is None or view.t is None:
            raise ValueError("Pose needs to be defined for the view before added to Q.")

        self.views.append(view)

    def get(self, index):
        return self.views[index]

    def get_all_exts(self):
        all_ext_mtx = None
        for i in range(len(self.views)):
            if all_ext_mtx is None:
                all_ext_mtx = self.get(i).get_external_mtx()
            else:
                all_ext_mtx = np.concatenate((all_ext_mtx, self.get(i).get_external_mtx()), axis = 0)
            
        return all_ext_mtx
