import argparse

import model.backbone as backbone
import metric.pairsampler as pair

def get_parser():
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    # parser.add_argument('--dataset',
    #                     choices=dict(cub200=dataset.CUB2011Metric,
    #                                  cars196=dataset.Cars196Metric,
    #                                  stanford=dataset.StanfordOnlineProductsMetric),
    #                     default=dataset.CUB2011Metric,
    #                     action=LookupChoices)

    parser.add_argument('--base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                     inception_v1bn=backbone.InceptionV1BN,
                                     resnet18=backbone.ResNet18,
                                     resnet50=backbone.ResNet50),
                        default=backbone.ResNet50,
                        action=LookupChoices)

    parser.add_argument('--teacher_base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                     inception_v1bn=backbone.InceptionV1BN,
                                     resnet18=backbone.ResNet18,
                                     resnet50=backbone.ResNet50),
                        default=backbone.ResNet50,
                        action=LookupChoices)

    parser.add_argument('--triplet_ratio', default=0, type=float)
    parser.add_argument('--dist_ratio', default=0, type=float)
    parser.add_argument('--angle_ratio', default=0, type=float)

    parser.add_argument('--dark_ratio', default=0, type=float)
    parser.add_argument('--dark_alpha', default=2, type=float)
    parser.add_argument('--dark_beta', default=3, type=float)

    parser.add_argument('--at_ratio', default=0, type=float)

    parser.add_argument('--triplet_sample',
                        choices=dict(random=pair.RandomNegative,
                                     hard=pair.HardNegative,
                                     all=pair.AllPairs,
                                     semihard=pair.SemiHardNegative,
                                     distance=pair.DistanceWeighted),
                        default=pair.DistanceWeighted,
                        action=LookupChoices)

    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')
    parser.add_argument('--embedding_size', default=128, type=int)

    parser.add_argument('--teacher_load', default=None, required=True)
    parser.add_argument('--teacher_l2normalize', choices=['true', 'false'], default='true')
    parser.add_argument('--teacher_embedding_size', default=128, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--data', default='data')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--iter_per_epoch', default=100, type=int)
    parser.add_argument('--lr_decay_epochs', type=int, default=[40, 60], nargs='+')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--load', default=None)
    return parser
