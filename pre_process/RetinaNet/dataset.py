import os

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir

        self.transfroms = transforms

        self.data_points = self._find_data_points()
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        data_point = self.data_points[idx]
        image_path = data_point['image_path']

        image = decode_image(image_path)
        target = self._parse_annotation(data_point)

        if self.transform:
            image, target = self.transform(image, target)

        data_point = {'image':image, 'target': target}

        return data_point
    
    def _find_data_points(self):
        image_dir = os.path.join(self.root_dir, 'images')
        label_dir = os.path.join(self.root_dir, 'labels')

        if not os.path.isdir(label_dir):
            error_message = (
                f"\n\nエラー: ラベルディレクトリが見つかりません: {label_dir}\n"
                "--------------------------------------------------------------------------\n"
                "考えられる原因と解決策:\n"
                "1. パスの指定が間違っている:\n"
                "   -> `root_dir`のパスが正しいか、ディレクトリ名が'labels'かを確認してください。\n\n"
                "2. データセットの形式が違う:\n"
                "   -> このデータセットは、アノテーションが単一ファイル(例: .json, .csv)にまとまっていませんか？\n"
                "   -> その場合、このデフォルトの読み込み処理は使えません。\n"
                "   -> 解決策: このクラスを継承した子クラスで、`_load_data_points`メソッドをデータセット形式に合わせてオーバーライドしてください。\n"
                "--------------------------------------------------------------------------"
            )
            raise FileNotFoundError(error_message)

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"エラー: 画像ディレクトリが見つかりません: {image_dir}")
        
        data_points = []
        image_filenames = sorted(os.listdir(image_dir))
        
        for img_name in image_filenames:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            label_name = os.path.splitext(img_name)[0] + self._get_label_extension()
            label_path = os.path.join(self.label_dir, label_name)
            
            if os.path.exists(label_path):
                image_path = os.path.join(self.image_dir, img_name)
                data_points.append({'image_path': image_path, 'label_path': label_path})
            else:
                print(f"⚠️ 警告: アノテーションファイルが見つかりません。スキップします: {label_path}")
        
        return data_points
    
    def _get_label_extension(self):
        """ラベルファイルの拡張子を返す。子クラスで必ず実装する。"""
        raise NotImplementedError("このメソッドは子クラスでオーバーライドしてください")
    
    def _parse_annotation(self, data_point):
        """
        アノテーションファイルを読み込んで整形する。
        座標は必ず'tv_tensors.BoundingBoxes'型に変形する。
        出力形式
        target = {
        "boxes": boxes,
        "labels": labels,
        }
        データセット形式ごとに違うので、子クラスで必ず実装する。
        """
        raise NotImplementedError("このメソッドは子クラスでオーバーライドしてください")