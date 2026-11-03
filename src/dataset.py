import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(train_path='train.csv', test_path='test.csv', val_size=0.1, random_state=42):
    """
    Đọc dữ liệu AG News (4 lớp: 1-World, 2-Sports, 3-Business, 4-Sci/Tech)
    và chia tập train/validation/test.
    """
    # Đọc dữ liệu
    train_df = pd.read_csv(train_path, names=['Class Index', 'Title', 'Description'], header=0)
    test_df = pd.read_csv(test_path, names=['Class Index', 'Title', 'Description'], header=0)
    
    # Xử lý các dòng nhãn lỗi/trùng tiêu đề
    train_df['Class Index'] = pd.to_numeric(train_df['Class Index'], errors='coerce')
    test_df['Class Index'] = pd.to_numeric(test_df['Class Index'], errors='coerce')
    train_df = train_df.dropna(subset=['Class Index'])
    test_df = test_df.dropna(subset=['Class Index'])
    
    # Chuyển nhãn về dạng 0, 1, 2, 3 (phù hợp với PyTorch / Transformer)
    train_df['label'] = train_df['Class Index'].astype(int) - 1
    test_df['label'] = test_df['Class Index'].astype(int) - 1
    
    # Ghép Title và Description thành 1 văn bản duy nhất
    train_df['text'] = train_df['Title'].fillna('') + " " + train_df['Description'].fillna('')
    test_df['text'] = test_df['Title'].fillna('') + " " + test_df['Description'].fillna('')
    
    # Chia tập train thành Train và Validation (Phân tầng theo nhãn)
    train_data, val_data = train_test_split(
        train_df[['text', 'label']], 
        test_size=val_size, 
        random_state=random_state, 
        stratify=train_df['label']
    )
    
    return train_data, val_data, test_df[['text', 'label']]
