## 2021-09-06  17:00 v1.0   添加GUI界面，实现音视频转字幕功能, 人声提取功能, 会议记录功能, 语言选择功能, 输出格式选择功能, 保存目录选择功能, 临时文件目录选择功能.当前英文和混合模型存在问题，功能暂无法使用。

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import configparser
import demucs.separate
from pydub import AudioSegment
import ffmpeg
import logging
import torch


# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from funasr import AutoModel

logging.info("程序开始运行")

cuda = 1
if not torch.cuda.is_available():
    cuda = 0
    logging.warning("CUDA不可用，将回退到CPU运行")
else:
    logging.info("CUDA 可用，将使用GPU运行")

# 读取配置文件
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")

# 从配置文件获取音频和视频文件扩展名
audio_extensions = [
    f".{ext.strip()}" for ext in config.get("file_types", "audio_extensions").split(",")
]
video_extensions = [
    f".{ext.strip()}" for ext in config.get("file_types", "video_extensions").split(",")
]

demucs_model_name = config.get("demucs", "model", fallback="mdx_extra")

funasr_model_zh = config.get("funasr", "funasr_model_zh")
funasr_vad_model_zh = config.get("funasr", "funasr_vad_model_zh")
funasr_punc_model_zh = config.get("funasr", "funasr_punc_model_zh")
funasr_spk_model_zh = config.get("funasr", "funasr_spk_model_zh")

funasr_model_en = config.get("funasr", "funasr_model_en")
funasr_vad_model_en = config.get("funasr", "funasr_vad_model_en")
funasr_punc_model_en = config.get("funasr", "funasr_punc_model_en")
funasr_spk_model_en = config.get("funasr", "funasr_spk_model_en")

funasr_model_mix = config.get("funasr", "funasr_model_mix")
funasr_vad_model_mix = config.get("funasr", "funasr_vad_model_mix")
funasr_punc_model_mix = config.get("funasr", "funasr_punc_model_mix")


try:
    demucs_batch_size = config.getint("demucs", "batch_size", fallback=4)
except ValueError:
    demucs_batch_size = 4  # 默认值

demucs_model_path = config.get("demucs", "model_path", fallback="")

os.environ["TORCH_HOME"] = demucs_model_path


# 封装错误处理函数
def handle_error(error_message, context=""):
    logging.error(f"{context}: {error_message}")
    messagebox.showerror("错误", f"{context}: {error_message}")


# 毫秒转换为分钟和秒的函数
def convert_milliseconds(milliseconds):
    total_seconds = milliseconds / 1000
    minutes = int(total_seconds // 60)
    seconds = round(total_seconds % 60, 2)
    return f"{minutes} 分 {seconds} 秒"


# 毫秒转换为 SRT 时间戳的函数
def ms_to_srt_timestamp(ms):
    hours = ms // (1000 * 60 * 60)
    ms %= 1000 * 60 * 60
    minutes = ms // (1000 * 60)
    ms %= 1000 * 60
    seconds = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


# FunASR 结果列表转换为 SRT 字幕的函数
def funasr_result_list_to_srt(res_list):
    srt_lines = []
    for i, res in enumerate(res_list, start=1):
        start_ms = res.get("start", 0)
        end_ms = res.get("end", 0)
        text = res.get("text", "")
        start_timestamp = ms_to_srt_timestamp(start_ms)
        end_timestamp = ms_to_srt_timestamp(end_ms)
        srt_lines.extend([str(i), f"{start_timestamp} --> {end_timestamp}", text, ""])
    return "\n".join(srt_lines)


# 处理段落结束的逻辑
def handle_paragraph_end(res_list, index, current_start, current_speaker, current_text):
    end_ms = res_list[index - 1].get("end", 0)
    start_timestamp = ms_to_srt_timestamp(current_start)
    end_timestamp = ms_to_srt_timestamp(end_ms)
    full_text = " ".join(current_text)
    return [f"{start_timestamp} - {end_timestamp}", f"{current_speaker}: {full_text}"]


# FunASR 结果列表转换为会议记录的函数，按段落处理，时间戳与内容分行，发言人前置
def funasr_result_list_to_meeting_record(res_list):
    if not res_list:
        return ""
    record_lines = []
    current_speaker = res_list[0].get("spk", "未知")
    current_start = res_list[0].get("start", 0)
    current_text = []
    for i, res in enumerate(res_list):
        speaker = res.get("spk", "未知")
        if speaker != current_speaker:
            record_lines.extend(
                handle_paragraph_end(
                    res_list, i, current_start, current_speaker, current_text
                )
            )
            current_speaker = speaker
            current_start = res.get("start", 0)
            current_text = [res.get("text", "")]
        else:
            current_text.append(res.get("text", ""))

    # 处理最后一段
    record_lines.extend(
        handle_paragraph_end(
            res_list, len(res_list), current_start, current_speaker, current_text
        )
    )

    return "\n".join(record_lines)


# 选择文件夹的函数
def select_folder(entry):
    folder = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, folder)


# 根据保存路径选项控制保存目录输入框状态
def update_srt_entry_state(srt_entry, srt_button, srt_dir_option_var):
    srt_dir_option = srt_dir_option_var.get()
    if srt_dir_option == 1:
        srt_entry.config(state=tk.DISABLED)
        srt_button.config(state=tk.DISABLED)
    else:
        srt_entry.config(state=tk.NORMAL)
        srt_button.config(state=tk.NORMAL)


def update_output_radio_state(meeting_record_radio, lang_var, output_format_var):
    lang_var_option = lang_var.get()
    if lang_var_option == "mix" or lang_var_option == "en":
        meeting_record_radio.config(state=tk.DISABLED)
        output_format_var.set("srt")
    else:
        meeting_record_radio.config(state=tk.NORMAL)
        output_format_var.set("srt")


# 使用 Demucs 进行人声提取的函数
def extract_vocals(input_file, output_dir):
    try:
        device = "cuda:0" if cuda else "cpu"
        model = demucs_model_name
        demucs.separate.main(
            [
                "--two-stems",
                "vocals",
                "-n",
                model,
                "--device",
                device,
                "--out",
                output_dir,
                input_file,
            ]
        )

        input_filename_without_ext = os.path.splitext(os.path.basename(input_file))[0]
        vocals_file = os.path.join(
            output_dir, model, input_filename_without_ext, "vocals.wav"
        )
        others_file = os.path.join(
            output_dir, model, input_filename_without_ext, "others.wav"
        )

        logging.info(f"提取的人声已保存为 {vocals_file}")
        return vocals_file, others_file

    except Exception as e:
        handle_error(f"人声提取失败: {str(e)}", context="extract_vocals")
        return input_file, None


# 将音频文件转换为 WAV 格式
def convert_audio_to_wav(input_file, temp_wav_file):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(temp_wav_file, format="wav")
        logging.info(f"音频转换为 WAV 成功，保存为 {temp_wav_file}")
        return temp_wav_file
    except Exception as e:
        handle_error(f"音频转换失败: {str(e)}", context="convert_audio_to_wav")
        return None


# 将视频文件转换为 WAV 格式
def convert_video_to_wav(input_file, temp_wav_file):
    try:
        (
            ffmpeg.input(input_file)
            .output(temp_wav_file, vn=None, acodec="pcm_s16le", ar=16000)
            .run(overwrite_output=True)
        )
        logging.info(f"音频提取并转换为 WAV 成功，保存为 {temp_wav_file}")
        return temp_wav_file
    except Exception as e:
        handle_error(f"音频提取失败: {e}", context="convert_video_to_wav")
        return None


# 将文件转换为 WAV 格式
def convert_to_wav(input_file, temp_dir):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    temp_wav_file = os.path.join(temp_dir, f"{base_name}_temp.wav")

    if any(input_file.lower().endswith(ext) for ext in audio_extensions):
        if not input_file.lower().endswith(".wav"):
            return convert_audio_to_wav(input_file, temp_wav_file)
        return input_file
    elif any(input_file.lower().endswith(ext) for ext in video_extensions):
        return convert_video_to_wav(input_file, temp_wav_file)
    else:
        handle_error(f"不支持的文件类型: {input_file}", context="convert_to_wav")
        return None


# 删除文件列表中的文件
def delete_files(file_list):
    for file in file_list:
        if file and os.path.exists(file):
            try:
                os.remove(file)
                logging.info(f"已删除文件: {file}")
            except Exception as e:
                handle_error(
                    f"删除文件 {file} 时出错: {str(e)}", context="delete_files"
                )


# 加载模型并生成结果
def load_model_and_generate(vocal_file, lang):
    #    try:

    if lang == "zh":
        params = {
            "model": funasr_model_zh,
            "vad_model": funasr_vad_model_zh,
            "punc_model": funasr_punc_model_zh,
            "spk_model": funasr_spk_model_zh,
            "disable_update": True,
            "device": "cuda:0" if cuda else "cpu",
        }
    elif lang == "en":
        params = {
            "model": funasr_model_en,
            "vad_model": funasr_vad_model_en,
            "punc_model": funasr_punc_model_en,
            "spk_model": funasr_spk_model_en,
            "disable_update": True,
            "device": "cuda:0" if cuda else "cpu",
        }
    elif lang == "mix":
        params = {
            "model": funasr_model_mix,
            "vad_model": funasr_vad_model_mix,
            "punc_model": funasr_punc_model_mix,
            "disable_update": True,
            "device": "cuda:0" if cuda else "cpu",
        }

    model = AutoModel(**params)
    res = model.generate(
        input=vocal_file,
        batch_size_s=30,
    )
    print(res)
    return res[0]["sentence_info"]


"""     except Exception as e:
        handle_error(f"模型生成结果失败: {str(e)}", context="load_model_and_generate")
        return None """


# 保存结果到文件
def save_result_to_file(sentence_info, output_format, base_name, srt_dir):
    if output_format == "srt":
        content = funasr_result_list_to_srt(sentence_info)
        file_extension = ".srt"
    else:
        content = funasr_result_list_to_meeting_record(sentence_info)
        file_extension = ".txt"

    output_file_path = os.path.join(srt_dir, f"{base_name}{file_extension}")
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_file_path
    except Exception as e:
        handle_error(f"保存结果到文件失败: {str(e)}", context="save_result_to_file")
        return None


# 处理单个文件的函数
def process_single_file(
    input_file,
    srt_dir_option,
    output_format,
    enable_vocal_extraction,
    temp_dir,
    srt_entry,
    lang,
):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    files_to_delete = []

    temp_wav_file = convert_to_wav(input_file, temp_dir)
    if not temp_wav_file:
        return

    if temp_wav_file != input_file:
        files_to_delete.append(temp_wav_file)

    vocal_file = temp_wav_file
    others_file = None
    if enable_vocal_extraction:
        vocal_file, others_file = extract_vocals(temp_wav_file, temp_dir)
        if vocal_file != temp_wav_file:
            files_to_delete.append(vocal_file)
        if others_file:
            files_to_delete.append(others_file)

    sentence_info = load_model_and_generate(vocal_file, lang)
    if not sentence_info:
        return

    if srt_dir_option == 1:
        srt_dir = os.path.dirname(input_file)
    else:
        srt_dir = srt_entry.get()

    output_file_path = save_result_to_file(
        sentence_info, output_format, base_name, srt_dir
    )

    # 删除临时文件
    if not input_file.lower().endswith(".wav"):
        delete_files(files_to_delete)

    if output_file_path:
        logging.info(f"文件已保存到: {output_file_path}")


# 获取文件夹内所有支持的文件
def get_supported_files(folder):
    supported_extensions = audio_extensions + video_extensions
    supported_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                supported_files.append(os.path.join(root, file))
    return supported_files


# 运行转换的函数
def run_conversion(
    file_entry,
    srt_dir_option_var,
    output_format_var,
    vocal_extraction_var,
    temp_entry,
    srt_entry,
    lang,
):
    input_folder = file_entry.get()
    srt_dir_option = srt_dir_option_var.get()
    output_format = output_format_var.get()
    enable_vocal_extraction = vocal_extraction_var.get()
    temp_dir = temp_entry.get()
    lang = lang.get()

    if not input_folder or not temp_dir:
        if srt_dir_option == 2 and not srt_entry.get():
            handle_error(
                "请选择输入文件夹、临时文件目录和保存目录", context="run_conversion"
            )
        else:
            handle_error("请选择输入文件夹和临时文件目录", context="run_conversion")
        return

    input_files = get_supported_files(input_folder)
    for input_file in input_files:
        process_single_file(
            input_file,
            srt_dir_option,
            output_format,
            enable_vocal_extraction,
            temp_dir,
            srt_entry,
            lang,
        )


# 创建 GUI 界面
def create_gui():
    root = tk.Tk()
    root.title("FunASR 音视频转字幕")

    # 使用 ttk 样式
    style = ttk.Style()
    style.theme_use("default")

    # 固定文本框和按钮的宽度
    entry_width = 50
    button_width = 10

    # 创建文件夹选择框和按钮
    file_label = ttk.Label(root, text="选择音视频所在文件夹:")
    file_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    file_entry = ttk.Entry(root, width=entry_width)
    file_entry.grid(row=0, column=1, padx=5, pady=5)
    file_button = ttk.Button(
        root, text="浏览", width=button_width, command=lambda: select_folder(file_entry)
    )
    file_button.grid(row=0, column=2, padx=5, pady=5)

    # 创建保存目录选择框和按钮
    srt_label = ttk.Label(root, text="选择保存目录（可选）:")
    srt_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    srt_entry = ttk.Entry(root, width=entry_width)
    srt_entry.grid(row=1, column=1, padx=5, pady=5)
    srt_button = ttk.Button(
        root, text="浏览", width=button_width, command=lambda: select_folder(srt_entry)
    )
    srt_button.grid(row=1, column=2, padx=5, pady=5)

    # 创建保存目录选择选项
    srt_dir_option_label = ttk.Label(root, text="保存文件路径选项:")
    srt_dir_option_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

    srt_dir_option_var = tk.IntVar()
    srt_dir_option_var.set(1)  # 默认选择保存到原始文件目录

    original_dir_radio = ttk.Radiobutton(
        root,
        text="保存到原始文件所在目录",
        variable=srt_dir_option_var,
        value=1,
        command=lambda: update_srt_entry_state(
            srt_entry, srt_button, srt_dir_option_var
        ),
    )
    original_dir_radio.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

    custom_dir_radio = ttk.Radiobutton(
        root,
        text="手动选择保存目录",
        variable=srt_dir_option_var,
        value=2,
        command=lambda: update_srt_entry_state(
            srt_entry, srt_button, srt_dir_option_var
        ),
    )
    custom_dir_radio.grid(
        row=2, column=1, padx=(5 + entry_width * 8, 5), pady=5, sticky=tk.W
    )

    # 初始化保存目录输入框状态
    update_srt_entry_state(srt_entry, srt_button, srt_dir_option_var)

    # 创建临时文件目录选择框和按钮
    temp_label = ttk.Label(root, text="选择临时文件存放目录:")
    temp_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
    temp_entry = ttk.Entry(root, width=entry_width)
    temp_entry.grid(row=3, column=1, padx=5, pady=5)
    temp_button = ttk.Button(
        root, text="浏览", width=button_width, command=lambda: select_folder(temp_entry)
    )
    temp_button.grid(row=3, column=2, padx=5, pady=5)

    # 创建输出格式选择框
    output_format_label = ttk.Label(root, text="选择输出格式:")
    output_format_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)

    output_format_var = tk.StringVar(root)
    output_format_var.set("srt")  # 默认选择 SRT 格式

    srt_radio = ttk.Radiobutton(
        root, text="SRT 字幕文件", variable=output_format_var, value="srt"
    )
    srt_radio.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)

    meeting_record_radio = ttk.Radiobutton(
        root, text="会议记录文件", variable=output_format_var, value="meeting_record"
    )
    meeting_record_radio.grid(
        row=5, column=1, padx=(5 + entry_width * 8, 5), pady=5, sticky=tk.W
    )

    lang_label = ttk.Label(root, text="选择输入的语言类型:")
    lang_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
    lang_var = tk.StringVar(root)
    lang_var.set("zh")  # 默认选择 zh中文

    lang_zh_radio = ttk.Radiobutton(
        root,
        text="中文zh",
        variable=lang_var,
        value="zh",
        command=lambda: update_output_radio_state(
            meeting_record_radio, lang_var, output_format_var
        ),
    )
    lang_zh_radio.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)

    lang_en_radio = ttk.Radiobutton(
        root,
        text="英文en",
        variable=lang_var,
        value="en",
        command=lambda: update_output_radio_state(
            meeting_record_radio, lang_var, output_format_var
        ),
    )
    lang_en_radio.grid(
        row=6, column=1, padx=(5 + entry_width * 4, 5), pady=5, sticky=tk.W
    )

    lang_mix_radio = ttk.Radiobutton(
        root,
        text="混合mix",
        variable=lang_var,
        value="mix",
        command=lambda: update_output_radio_state(
            meeting_record_radio, lang_var, output_format_var
        ),
    )
    lang_mix_radio.grid(
        row=6, column=1, padx=(5 + entry_width * 8, 5), pady=5, sticky=tk.W
    )

    # 创建是否启用人声提取优化功能的选择框
    vocal_extraction_label = ttk.Label(root, text="启用人声提取优化功能:这个比较耗时")
    vocal_extraction_label.grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)

    vocal_extraction_var = tk.IntVar()
    vocal_extraction_checkbox = ttk.Checkbutton(root, variable=vocal_extraction_var)
    vocal_extraction_checkbox.grid(row=7, column=1, padx=5, pady=5, sticky=tk.W)

    # 创建确定按钮
    convert_button = ttk.Button(
        root,
        text="确定",
        command=lambda: run_conversion(
            file_entry,
            srt_dir_option_var,
            output_format_var,
            vocal_extraction_var,
            temp_entry,
            srt_entry,
            lang_var,
        ),
    )
    convert_button.grid(row=7, column=1, padx=5, pady=20)

    return root


if __name__ == "__main__":
    root = create_gui()
    root.mainloop()
