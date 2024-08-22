from praatio import tgio
from funsound.utils import *
from funsound.compute_mer import text_clean
SR = 16000

def merge_lines(lines, k, t):
    merged_lines = []
    current_line = None

    for line in lines:
        ts, te, role, text = line['start'], line['end'], line['role'], line['text']
        
        if current_line is None:
            current_line = {'start': ts, 'end': te, 'role': role, 'text': text}
            continue

        # 检查是否应该合并
        if current_line['role'] == role and (ts - current_line['end']) <= k:
            new_end = max(te, current_line['end'])
            new_text = current_line['text'] + ' ' + text
            if (new_end - current_line['start']) <= t:
                current_line['end'] = new_end
                current_line['text'] = new_text
            else:
                merged_lines.append(current_line)
                current_line = {'start': ts, 'end': te, 'role': role, 'text': text}
        else:
            merged_lines.append(current_line)
            current_line = {'start': ts, 'end': te, 'role': role, 'text': text}

    if current_line is not None:
        merged_lines.append(current_line)

    return merged_lines



def make_sentences(audio_file,grid_file,outdir):
    task = get_utt(audio_file)
    outdir = mkdir(os.path.join(outdir,task),reset=True)
    
    tg = tgio.openTextgrid(grid_file)
    tier = tg.tierDict['text']
    lines = []
    for interval in tier.entryList:
        ts, te, label = interval
        if not label:continue
        print(label)
        role = label.split()[-1]
        text = "".join(label.split()[:-1])
        text = text_clean(text)
        assert role in ["TR","ST","MX"]
        line = {'start':round(ts,2),
                        'end':round(te,2),
                        'dur':round(te-ts,2),
                        'role':role,
                        'text':text}
        lines.append(line)

    audio_data = read_audio_file(audio_file)
    audio_len = len(audio_data) / SR
    lines = merge_lines(lines,2,20)
    for i,line in enumerate(lines):
        ts,te = int(SR*line['start']), int(SR*line['end'])
        role, text = line['role'], line['text']
        if len(text)<2:continue
        utt = "%s_%s_%08d"%(task,role,i)
        
        mkdir(f"{outdir}/{role}")
        seg_audio_file = os.path.join(f"{outdir}/{role}", f'{utt}.wav')
        seg_label_file = os.path.join(f"{outdir}/{role}", f'{utt}.txt')
        seg_audio_data = audio_data[ts:te]
        save_wavfile(seg_audio_file,seg_audio_data)
        with open(seg_label_file,'wt',encoding='utf-8') as f:
            print(f"{line['text']}",file=f)

if __name__ == "__main__":

    make_sentences(audio_file="dataset/道德_自然灾害.mp3",
                   grid_file="dataset/道德_自然灾害.TextGrid",
                   outdir="dataset")