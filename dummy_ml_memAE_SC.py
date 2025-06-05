import numpy as np
import joblib
import os

def create_dummy_ml_memae_sc_chunk(chunk_file_path, num_samples=128, clip_len=5, motion_channels=2, appearance_channels=3, patch_size=32):
    """
    ml_memAE_sc_train.py 用のダミーデータチャンクファイルを作成する関数。
    extract_samples.py が生成する chunked_samples_xx.pkl の形式を模倣する。
    """
    
    # 'motion' データを作成 (ML_MemAE_SC が主に使用)
    # 各サンプルは [clip_len, motion_channels, patch_size, patch_size] の形状
    # 例: [5, 2, 32, 32]
    motion_data = np.random.rand(num_samples, clip_len, patch_size, patch_size, motion_channels).astype(np.float32)
    
    # 他のキーも形式だけ合わせてダミーデータを作成
    appearance_data = np.random.rand(num_samples, clip_len,  patch_size, patch_size, appearance_channels,).astype(np.float32)
    bbox_data = np.random.rand(num_samples, 4).astype(np.float32)  # [x1, y1, x2, y2]
    
    # pred_frame は extract_samples.py では frameRange[-num_predicted_frame:] となっている。
    # Chunked_sample_dataset の __getitem__ では pred_frame そのものが返される。
    # eval.py などで pred_frame_test[i][-1].item() のように使われることがあるため、
    # (num_samples, 1) のような形状でダミーのフレームインデックスを入れておく。
    pred_frame_data = np.random.randint(0, 1000, size=(num_samples, 1)).astype(np.int32) 
    sample_id_data = np.arange(num_samples).astype(np.int32)

    chunked_samples = {
        "sample_id": sample_id_data,
        "appearance": appearance_data,
        "motion": motion_data,
        "bbox": bbox_data,
        "pred_frame": pred_frame_data 
    }
    
    # 保存先のディレクトリがなければ作成
    save_dir = os.path.dirname(chunk_file_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    joblib.dump(chunked_samples, chunk_file_path)
    print(f"Dummy chunk file for ml_memAE_sc_train.py created at: {chunk_file_path}")
    print(f"  Each 'motion' sample shape: {motion_data[0].shape}")

if __name__ == '__main__':
    # --- 設定項目 ---
    dummy_data_root = "./dummy"  # ダミーデータのルートディレクトリ
    dataset_name = "ped2"  # データセット名 (パスの構成に使用)
    mode = 'testing'
    num_dummy_samples = 256  # 1チャンクあたりのサンプル数 (バッチサイズに合わせると良いかも)
    num_dummy_chunks = 10     # 作成するチャンクファイルの数
    
    # STCのパラメータ (extract_samples.py や ml_memAE_sc_cfg.yaml を参照)
    stc_clip_len = 5  # 通常 dataset.context_frame_num + 1 (HF2-VADでは通常4+1=5)
                      # ただし、ml_memAE_sc_train.pyではnum_flows=1として最後の1フレームのフローを使う
    stc_motion_channels = 2
    stc_appearance_channels = 3 # RGBなので3
    stc_patch_size = 32
    # --- 設定項目ここまで ---

    training_chunked_samples_dir = os.path.join(dummy_data_root, dataset_name, mode, "chunked_samples")
    
    for i in range(num_dummy_chunks):
        chunk_filename = f"chunked_samples_{i:02d}.pkl"
        full_chunk_path = os.path.join(training_chunked_samples_dir, chunk_filename)
        create_dummy_ml_memae_sc_chunk(
            chunk_file_path=full_chunk_path,
            num_samples=num_dummy_samples,
            clip_len=stc_clip_len,
            motion_channels=stc_motion_channels,
            appearance_channels=stc_appearance_channels,
            patch_size=stc_patch_size
        )
    
    print("\n--- Next Steps ---")
    print(f"1. Update 'training_chunked_samples_dir' in your 'ml_memAE_sc_cfg.yaml' to: {training_chunked_samples_dir}")
    print(f"2. Update 'testing_chunked_samples_file' in 'ml_memAE_sc_cfg.yaml' or relevant eval script if needed (though this script is for training).")
    print(f"3. Ensure parameters in 'ml_memAE_sc_cfg.yaml' (like motion_channels, num_flows) are consistent with the dummy data structure if you modify it.")