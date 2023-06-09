WANDB_PROJECT = "mlops-course-001-GestureRecog"
ENTITY = None # set this to team name if working in a team
BDD_CLASSES = {0:'Left_Swipe', 1:'Right_Swipe', 2:'Stop', 3:'Thumbs_Down', 4:'Thumbs_Up'}
RAW_DATA_AT = 'Project_data'
PROCESSED_DATA_AT = 'Project_data_split'



zip_file_path = './Project_data.zip'
unzipped_file_path = './Project_data'
train_image_path = './Project_data/train'      # path to train image set
test_image_path = './Project_data/test'          # path to test image set
train_csv_path = './Project_data/train.csv'      # path to train csv 
test_csv_path = './Project_data/test.csv'          # path to test csv 
seed = 0