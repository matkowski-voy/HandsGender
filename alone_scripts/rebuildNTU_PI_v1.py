import pandas as pd
import os
import shutil

def prepare_db():

    gender_csv_file_name = 'NTU-PI-v1-gender.csv'
    dst_db_name = 'NTU-PI-v1-gender'
    src_db_name = 'NTU-PI-v1'
    
    data = pd.read_csv(gender_csv_file_name,header=None)
    
    if not os.path.exists(dst_db_name):
        os.mkdir(dst_db_name)
        os.mkdir(os.path.join(dst_db_name,'imgOrg'))
        os.mkdir(os.path.join(dst_db_name,'imgOrg','male'))
        os.mkdir(os.path.join(dst_db_name,'imgOrg','female'))
        os.mkdir(os.path.join(dst_db_name,'imgMask'))
        os.mkdir(os.path.join(dst_db_name,'imgMask','male'))
        os.mkdir(os.path.join(dst_db_name,'imgMask','female'))
        
        # only using landmarks because NTU-PI-v1 does not provide ROIs
        if os.path.exists(os.path.join(src_db_name,'flip','ROImarked')):
            os.mkdir(os.path.join(dst_db_name,'ROImarked'))
            os.mkdir(os.path.join(dst_db_name,'ROImarked','male'))
            os.mkdir(os.path.join(dst_db_name,'ROImarked','female'))
    
    for i in range(len(data)):
        
        subject_id = data.iloc[i,0][0:data.iloc[i,0].find('-')]
        subject_gender = data.iloc[i,1]
        
        # imgOrg
        img_type = 'imgOrg'
        if not os.path.exists(os.path.join(dst_db_name,img_type,subject_gender,subject_id)):
            os.mkdir(os.path.join(dst_db_name,img_type,subject_gender,subject_id))
            
        if os.path.exists(os.path.join(src_db_name,'flip',img_type,'train',data.iloc[i,0])):
            shutil.copyfile(os.path.join(src_db_name,'flip',img_type,'train',data.iloc[i,0]), \
                            os.path.join(dst_db_name,img_type,subject_gender,subject_id,data.iloc[i,0]))
        
        if os.path.exists(os.path.join(src_db_name,'flip',img_type,'test',data.iloc[i,0])):
            shutil.copyfile(os.path.join(src_db_name,'flip',img_type,'test',data.iloc[i,0]), \
                            os.path.join(dst_db_name,img_type,subject_gender,subject_id,data.iloc[i,0]))
                
        # imgMask
        if not os.path.exists(os.path.join(dst_db_name,'imgMask',subject_gender,subject_id)):
            os.mkdir(os.path.join(dst_db_name,'imgMask',subject_gender,subject_id))
        
        if os.path.exists(os.path.join(src_db_name,'imgMask',data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpg')):
            
            shutil.copyfile(os.path.join(src_db_name,'imgMask',data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpg'), \
                            os.path.join(dst_db_name,'imgMask',subject_gender,subject_id,data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpg'))
        
        if os.path.exists(os.path.join(src_db_name,'imgMask',data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpeg')):
            
            shutil.copyfile(os.path.join(src_db_name,'imgMask',data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpeg'), \
                            os.path.join(dst_db_name,'imgMask',subject_gender,subject_id,data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpeg'))
    
        # ROImarked
        img_type = 'ROImarked'
        if os.path.exists(os.path.join(src_db_name,'flip','ROImarked')):
            if not os.path.exists(os.path.join(dst_db_name,img_type,subject_gender,subject_id)):
                os.mkdir(os.path.join(dst_db_name,img_type,subject_gender,subject_id))
                
            if os.path.exists(os.path.join(src_db_name,'flip',img_type,'train',data.iloc[i,0])):
                shutil.copyfile(os.path.join(src_db_name,'flip',img_type,'train',data.iloc[i,0]), \
                                os.path.join(dst_db_name,img_type,subject_gender,subject_id,data.iloc[i,0]))
            
            if os.path.exists(os.path.join(src_db_name,'flip',img_type,'test',data.iloc[i,0])):
                shutil.copyfile(os.path.join(src_db_name,'flip',img_type,'test',data.iloc[i,0]), \
                                os.path.join(dst_db_name,img_type,subject_gender,subject_id,data.iloc[i,0]))