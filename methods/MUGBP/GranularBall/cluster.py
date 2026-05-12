
import torch
import torch.nn as nn

from .cluster2 import GBNR

class gbcluster(nn.Module):

    def __init__(self,args,data):
        super(gbcluster, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, args, features, labels,select):
        self.device = features.device
        if select == False:
            a_purity = args.purity_train
        else:
            a_purity = args.purity_get_ball
        original_target = labels.to(self.device)
        index1 = torch.arange(len(labels), device=self.device)

        label_features = torch.cat((original_target.reshape(-1, 1), features.to(self.device)), dim=1)

        out = torch.cat((index1.reshape(-1, 1), label_features ), dim=1)
        out = torch.cat((original_target.reshape(-1, 1), out), dim=1)
        pur_tensor = torch.full((out.size(0), 1), a_purity, device=self.device)
        out = torch.cat((pur_tensor, out), dim=1)
        self.center, self.labels, self.radius = GBNR.apply(args, out, select)

        gb_centroids =self.center
        gb_radii=self.radius
        gb_labels=self.labels

        return  gb_centroids, gb_radii,gb_labels
