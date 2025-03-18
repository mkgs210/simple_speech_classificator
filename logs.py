import os
import shutil
from glob import glob

def reorganize_logs(source_dir):
    # Находим все файлы событий
    event_files = glob(os.path.join(source_dir, "**", "events.*"), recursive=True)
    
    for event_file in event_files:
        version_dir = os.path.dirname(event_file)
        version = os.path.basename(version_dir)
        
        # Создаем новую структуру директорий
        # Например: tb_logs_reorganized/groups_2_mels_20/version_0/
        if "version_" in version:
            version_num = int(version.split("_")[1])
            groups = [2, 4, 8, 16][version_num // 3]
            n_mels = [20, 40, 80][version_num % 3]
            
            new_dir = os.path.join("tb_logs_reorganized", 
                                 f"groups_{groups}_mels_{n_mels}", 
                                 version)
            
            # Создаем директорию и копируем файл
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy2(event_file, new_dir)

    print("Логи реорганизованы в tb_logs_reorganized/")

if __name__ == "__main__":
    source_dir = "tb_logs_[2, 4, 8, 16]_[20, 40, 80]/SpeechCommands"
    reorganize_logs(source_dir)