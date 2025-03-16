"""Datasets package."""

__all__ = []

__datasets__:  dict =   {
                            "jrc_cos7-1a":                  {
                                                                "name":         "Wild-type, interphase COS-7 cell (ATCC CRL-1651)",
                                                                "voxel_size":   (4, 4, 4),
                                                                "dimensions":   (25400, 6000, 4056),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_cos7-1a"
                                                            },
                            
                            "jrc_cos7-1b":                  {
                                                                "name":         "Wild-type, interphase COS-7 cell (ATCC CRL-1651)",
                                                                "voxel_size":   (4, 4, 5.08),
                                                                "dimensions":   (22200, 6800, 2844.8),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_cos7-1b"
                                                            },
                            
                            "jrc_ctl-id8-1":                {
                                                                "name":         "Killer T-Cell attacking cancer cell",
                                                                "voxel_size":   (4, 4, 3.48),
                                                                "dimensions":   (74, 13, 42),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_ctl-id8-1"
                                                            },
                            
                            "jrc_fly-mb-1a":                {
                                                                "name":         "Drosophila mushroom body",
                                                                "voxel_size":   (4, 4, 4),
                                                                "dimensions":   (44020, 37144, 38016),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_fly-mb-1a"
                                                            },
                            
                            "jrc_fly-vnc-1":                {
                                                                "name":         "Drosophila ventral nerve cord (VNC)",
                                                                "voxel_size":   (4, 4, 4.72),
                                                                "dimensions":   (40236, 20492, 16175.44),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_fly-vnc-1"
                                                            },
                            
                            "jrc_hela-2":                   {
                                                                "name":         "Interphase HeLa cell",
                                                                "voxel_size":   (4, 4, 5.24),
                                                                "dimensions":   (48, 6, 33),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_hela-2"
                                                            },
                            
                            "jrc_hela-3":                   {
                                                                "name":         "Interphase HeLa cell",
                                                                "voxel_size":   (4, 4, 3.24),
                                                                "dimensions":   (50, 4, 39),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_hela-3"
                                                            },
                            
                            "jrc_jurkat-1":                 {
                                                                "name":         "Immortalized T-Cells (Jurkat)",
                                                                "voxel_size":   (4, 4, 3.44),
                                                                "dimensions":   (40, 12, 29),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_jurkat-1"
                                                            },
                            
                            "jrc_marcophage-2":             {
                                                                "name":         "Macrophage cell",
                                                                "voxels":       (4, 4, 3.36),
                                                                "dimensions":   (40, 8, 37),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_marcophage-2"
                                                            },
                            
                            "jrc_mus-heart-1":              {
                                                                "name":         "P7 Mouse heart",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (163232, 151688, 135872),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_mus-heart-1"
                                                            },
                            
                            "jrc_mus-kidney":               {
                                                                "name":         "Mouse Kidney",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (98.3, 63.6, 177.6),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_mus-kidney"
                                                            },
                            
                            "jrc_mus-kidney-3":             {
                                                                "name":         "P7 Mouse Kidney",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (143360, 120784, 163072),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_mus-kidney-3"
                                                            },
                            
                            "jrc_mus-kidney-glomerulus-2":  {
                                                                "name":         "Mouse kidney glomerulus",
                                                                "voxels":       (4, 4, 4),
                                                                "dimensions":   (18984, 24608, 26536),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_mus-kidney-glomerulus-2"
                                                            },
                            
                            "jrc_mus-liver":                {
                                                                "name":         "Mouse Liver",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (102, 101.8, 71.5),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_mus-liver"
                                                            },
                            
                            "jrc_mus-liver-3":              {
                                                                "name":         "P7 Mouse liver",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (145320, 114976, 144256),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_mus-liver-3"
                                                            },
                            
                            "jrc_mus-liver-zon-1":          {
                                                                "name":         "Mouse liver acinus",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (188808, 171608, 397160),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_mus-liver-zon-1"
                                                            },
                            
                            "jrc_mus-liver-zon-2":          {
                                                                "name":         "Mouse liver acinus",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (210000, 155728, 426688),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_mus-liver-zon-2"
                                                            },
                            
                            "jrc_mus-nacc-1":               {
                                                                "name":         "Mouse nucleus accumbens",
                                                                "voxels":       (4, 4, 4),
                                                                "dimensions":   (9444, 9768, 2252),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_mus-nacc-1"
                                                            },
                            
                            "jrc_sum159-1":                 {
                                                                "name":         "Immortalized breast cancer cell (SUM159)",
                                                                "voxels":       (4, 4, 4.56),
                                                                "dimensions":   (64, 11, 35),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_sum159-1"
                                                            },
                            
                            "jrc_sum159-4":                 {
                                                                "name":         "Immortalized breast cancer cell (SUM159)",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (95000, 8496, 47800),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_sum159-4"
                                                            },
                            
                            "jrc_ut21-1413-003":            {
                                                                "name":         "Immortalized breast cancer cell (SUM159)",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (121192, 91224, 130416),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_ut21-1413-003"
                                                            },
                            
                            "jrc_zf-cardiac-1":             {
                                                                "name":         "7 Day post fertilization larval zebrafish",
                                                                "voxels":       (8, 8, 8),
                                                                "dimensions":   (165032, 158512, 321216),
                                                                "url":          "https://www.openorganelle.org/datasets/jrc_zf-cardiac-1"
                                                            }
                        }