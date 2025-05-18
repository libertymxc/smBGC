from torch_geometric.data import Data
from data.dataset_delete_five import get_ids, get_subject_label, transform_dataset


def get_y(dataset: [Data]):
    """
    Get the y values from a list of Data objects.
    """
    y = []
    for d in dataset:
        y.append(d.y.item())
    return y



if __name__ == '__main__':
    subject_IDs = get_ids()
    print(subject_IDs)
    labels = get_subject_label(subject_IDs, label="Group")
    FCN_dataset = transform_dataset(subject_IDs,kind="FCN",label=labels,droprate1=0.4,droprate2=0.4)
    print(get_y(FCN_dataset))