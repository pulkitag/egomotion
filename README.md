# egomotion

Dependencies
-------------
- [transforms3d](https://github.com/matthew-brett/transforms3d)
- [pycaffe_utils](https://github.com/pulkitag/pycaffe-utils)
- [geopy] (https://github.com/geopy/geopy)

Creating an Experiment
----------------------

- Pose experiments in my_exp_pose, my_exp_pose_v2
- Patch matching experiments in my_exp_ptch, my_exp_ptch_v2

Each function in these folders defines an experiment. See the files for an examples.
For instance one can use

```python
import my_exp_pose_v2 as mepo2
import street_utils as su
prms, cPrms = mepo2.smallnetv5_fc5_pose_euler_crp192_rawImSz256_lossl1()
```

If it is required to setup the training/testing data use:
```python
import street_labels_to_window as slw
#Make the window files per folder
#isForceWrite - set it to true if files need to be recreated
slw.make_window_files_geo_folders(prms, isForceWrite=False)
#Combine all the window files together
#setName: 'train', 'val', 'test'
su.make_combined_window_file(prms, setName='train')
```
the window files are now ready for running the experiments! 


