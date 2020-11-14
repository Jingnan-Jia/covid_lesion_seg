import argparse

parser = argparse.ArgumentParser(prog="MS-Net",
                                 description='Multi-scale Multi-task Semi-supervised network for Segmentation',
                                 epilog="If you need any help, contact jiajingnan2222@gmail.com")
# model_choices = ["net_lobe", "net_vessel", "net_recon", "net_lesion"]
parser.add_argument('-model_names', '--model_names', help='model names', type=str,
                    default='net_lesion-net_recon')
parser.add_argument('-main_model', '--main_model', help='main model is trained more frequently', type=str,
                    default='net_lesion')

parser.add_argument('-adaptive_lr', '--adaptive_lr', help='adaptive learning rate', type=int, default=1)
parser.add_argument('-attention', '--attention', help='attention loss', type=int, default=0)
parser.add_argument('-fn', '--feature_number', help='Number of initial of conv channels', type=int, default=32)
parser.add_argument('-bn', '--batch_norm', help='Set Batch Normalization', type=int, default=0)
parser.add_argument('-dr', '--dropout', help='Set Dropout', type=int, default=1)
parser.add_argument('-ptch_sz', '--ptch_sz',  type=int, default=192,
                    help='patch size for x, y axis, for 3D and 2D images')
parser.add_argument('-ptch_z_sz', '--ptch_z_sz', type=int, default=16,
                    help='patch size along z axis, be ignored for 2D images')
parser.add_argument('-batch_size', '--batch_size', help='batch_size', type=int, default=1)
parser.add_argument('-pps', '--patches_per_scan', help='patches_per_scan', type=int, default=3)
parser.add_argument('-p_middle', '--p_middle', help='sample in the middle parts', type=float, default=0)
parser.add_argument('-step_nb', '--step_nb', help='training step', type=int, default=240001)
parser.add_argument('-monitor_period', '--monitor_period', help='monitor_period', type=int, default=510)
parser.add_argument('-valid_period', '--valid_period', help='valid_period', type=int, default=510)
parser.add_argument('-cntd_pts', '--cntd_pts', help='connected parts for postprocessing', type=int, default=0)


parser.add_argument('-u_v', '--u_v', help='U-Net or V-Net', choices=["v", "u"], type=str, default='v')
parser.add_argument('-fat', '--fat', help='focus_alt_train', type=int, default=1)
parser.add_argument('-pad', '--pad', help='padding number outside original image', type=int, default=0)

# learning rate, normally lr of main model is greater than others
parser.add_argument('-lr_ls', '--lr_ls', type=float, default=0.00001)
parser.add_argument('-lr_lb', '--lr_lb', type=float, default=0.00001)
parser.add_argument('-lr_vs', '--lr_vs', type=float, default=0.00001)
parser.add_argument('-lr_aw', '--lr_aw', type=float, default=0.00001)
parser.add_argument('-lr_lu', '--lr_lu', type=float, default=0.00001)
parser.add_argument('-lr_rc', '--lr_rc', type=float, default=0.00001)

parser.add_argument('-lr_afd', '--lr_rc', type=float, default=0.00001)

# Number of Deep Supervisors
parser.add_argument('-ds_ls', '--ds_ls', type=int, default=2)
parser.add_argument('-ds_lb', '--ds_lb', type=int, default=0)
parser.add_argument('-ds_vs', '--ds_vs', type=int, default=0)
parser.add_argument('-ds_aw', '--ds_aw', type=int, default=0)
parser.add_argument('-ds_lu', '--ds_lu', type=int, default=0)
parser.add_argument('-ds_rc', '--ds_rc', type=int, default=0)

# aux output, it is fissure for lobe
parser.add_argument('-ao_ls', '--ao_ls', type=int, default=0)
parser.add_argument('-ao_lb', '--ao_lb', type=int, default=0)
parser.add_argument('-ao_vs', '--ao_vs', type=int, default=0)
parser.add_argument('-ao_aw', '--ao_aw', type=int, default=0)
parser.add_argument('-ao_lu', '--ao_lu', type=int, default=0)
parser.add_argument('-ao_rc', '--ao_rc', type=int, default=0)

# target spacing along (x, y) and z, format: m_n
parser.add_argument('-tsp_ls', '--tsp_ls', type=str, default='1.25_5')
parser.add_argument('-tsp_lb', '--tsp_lb', type=str, default='1.25_5')
parser.add_argument('-tsp_vs', '--tsp_vs', type=str, default='1.25_5')
parser.add_argument('-tsp_aw', '--tsp_aw', type=str, default='1.25_5')
parser.add_argument('-tsp_lu', '--tsp_lu', type=str, default='1.25_5')
parser.add_argument('-tsp_rc', '--tsp_rc', type=str, default='1.25_5')

# target size along (x, y) and z, format: m_n, --tsp will override these args.
parser.add_argument('-tsz_ls', '--tsz_ls', type=str, default='0_0')
parser.add_argument('-tsz_lb', '--tsz_lb', type=str, default='0_0')
parser.add_argument('-tsz_vs', '--tsz_vs', type=str, default='0_0')
parser.add_argument('-tsz_aw', '--tsz_aw', type=str, default='0_0')
parser.add_argument('-tsz_lu', '--tsz_lu', type=str, default='0_0')
parser.add_argument('-tsz_rc', '--tsz_rc', type=str, default='0_0')

# input outpt setting
io_choices = ["2_in_2_out", "1_in_low_1_out_low", "1_in_hgh_1_out_hgh", "2_in_1_out_low", "2_in_1_out_hgh"]
parser.add_argument('-ls_io', '--ls_io', type=str, choices=io_choices, default="1_in_low_1_out_low")
parser.add_argument('-lb_io', '--lb_io', type=str, choices=io_choices, default="1_in_low_1_out_low")
parser.add_argument('-vs_io', '--vs_io', type=str, choices=io_choices, default="1_in_low_1_out_low")
parser.add_argument('-aw_io', '--aw_io', type=str, choices=io_choices, default="1_in_low_1_out_low")
parser.add_argument('-lu_io', '--lu_io', type=str, choices=io_choices, default="1_in_low_1_out_low")
parser.add_argument('-rc_io', '--rc_io', type=str, choices=io_choices, default="1_in_low_1_out_low")

# number of training images, 0 means "all"
parser.add_argument('-ls_tr_nb', '--ls_tr_nb', help='rc_tr_nb', type=int, default=0)
parser.add_argument('-lb_tr_nb', '--lb_tr_nb', help='lb_tr_nb', type=int, default=0)
parser.add_argument('-vs_tr_nb', '--vs_tr_nb', help='vs_tr_nb', type=int, default=0)
parser.add_argument('-aw_tr_nb', '--aw_tr_nb', help='aw_tr_nb', type=int, default=0)
parser.add_argument('-lu_tr_nb', '--lu_tr_nb', help='lu_tr_nb', type=int, default=0)
parser.add_argument('-rc_tr_nb', '--rc_tr_nb', help='rc_tr_nb', type=int, default=0)

# number of training images, 0 means "all"
parser.add_argument('-ls_sub_dir', '--ls_sub_dir', help='rc_sub_dir', type=str, default='Covid_lesion')
parser.add_argument('-lb_sub_dir', '--lb_sub_dir', help='lb_sub_dir', type=str, default='GLUCOLD')
parser.add_argument('-vs_sub_dir', '--vs_sub_dir', help='vs_sub_dir', type=str, default='SSc')
parser.add_argument('-aw_sub_dir', '--aw_sub_dir', help='aw_sub_dir', type=str, default='None')
parser.add_argument('-lu_sub_dir', '--lu_sub_dir', help='lu_sub_dir', type=str, default='None')
parser.add_argument('-rc_sub_dir', '--rc_sub_dir', help='rc_sub_dir', type=str, default='LUNA16')

# name of loaded trained model for single-task net
parser.add_argument('-ld_ls', '--ld_ls', type=str, default='None')
parser.add_argument('-ld_lb', '--ld_lb', type=str, default='None')
parser.add_argument('-ld_vs', '--ld_vs', type=str, default='None')
parser.add_argument('-ld_aw', '--ld_aw', type=str, default='None')
parser.add_argument('-ld_lu', '--ld_lu', type=str, default='None')
parser.add_argument('-ld_rc', '--ld_rc', type=str, default='None')

# name of loaded trained model for integrated-task net
parser.add_argument('-ld_itgt_ls_rc', '--ld_itgt_ls_rc', type=str, default='None')
parser.add_argument('-ld_itgt_lb_rc', '--ld_itgt_lb_rc', type=str, default='None')
parser.add_argument('-ld_itgt_vs_rc', '--ld_itgt_vs_rc', type=str, default='None')
parser.add_argument('-ld_itgt_lu_rc', '--ld_itgt_lu_rc', type=str, default='None')
parser.add_argument('-ld_itgt_aw_rc', '--ld_itgt_aw_rc', type=str, default='None')

args = parser.parse_args()

