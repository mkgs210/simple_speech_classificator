from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        """
        Initialization of Wav2Vec2Decoder class
        
        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None
        
        self.ind2char = {0: '<pad>', 1: '<s>', 2: '</s>', 3: '<unk>', 4: '|', 5: 'E', 6: 'T', 7: 'A', 8: 'O', 9: 'N', 10: 'I', 11: 'H', 12: 'S', 13: 'R', 14: 'D', 15: 'L', 16: 'U', 17: 'M', 18: 'W', 19: 'C', 20: 'F', 21: 'G', 22: 'Y', 23: 'P', 24: 'B', 25: 'V', 26: 'K', 27: "'", 28: 'X', 29: 'J', 30: 'Q', 31: 'Z'}
        self.pad_token = 0
        self.space_token = 4

    def ctc_decode(self, inds):
        decoded = []
        last_char_ind = self.pad_token 
        for ind in inds:
            ind = int(ind)
            if last_char_ind == ind or ind == self.pad_token:
                continue
            elif ind == self.space_token:
                if last_char_ind != self.space_token:
                    decoded.append(' ')
            elif last_char_ind != self.pad_token:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind
        return ''.join(decoded)

    def greedy_decode(self, logits: torch.Tensor) -> str:
        text = self.ctc_decode([rec.argmax(-1).numpy() for rec in logits])
        return text

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        log_probs = torch.log_softmax(logits, dim=-1)
        beams = [(tuple(), self.blank_token_id, 0.0)]
        
        for t in range(len(log_probs)):
            t_log_probs = log_probs[t].numpy()
            candidates = []
            
            for prefix, last_token, log_p in beams:
                new_log_p = log_p + t_log_probs[last_token]
                candidates.append((prefix, last_token, new_log_p))
                
                for c in range(len(t_log_probs)):
                    if c != last_token:
                        new_prefix = prefix + (last_token,) if last_token != self.blank_token_id else prefix
                        candidates.append((new_prefix, c, log_p + t_log_probs[c]))
            
            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = candidates[:self.beam_width]
        
        final_beams = []
        for prefix, last_token, log_p in beams:
            if last_token != self.blank_token_id:
                prefix = prefix + (last_token,)
            final_beams.append((list(prefix), log_p))
        
        final_beams.sort(key=lambda x: x[1], reverse=True)
        
        if return_beams:
            return final_beams
        else:
            best_hypothesis = self.ctc_decode(final_beams[0][0])
            return best_hypothesis

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        log_probs = torch.log_softmax(logits, dim=-1)
        beams = [(tuple(), self.blank_token_id, 0.0, 0.0, "", 0.0)]
        
        for t in range(len(log_probs)):
            t_log_probs = log_probs[t].numpy()
            candidates = []
            
            for prefix, last_token, ac_score, lm_score, text, _ in beams:
                # Обработка повтора текущего токена
                new_ac_score = ac_score + t_log_probs[last_token]
                new_total_score = new_ac_score + self.alpha * lm_score + \
                                self.beta * len(text.split())
                candidates.append((prefix, last_token, new_ac_score, lm_score, text, new_total_score))
                
                # Обработка новых токенов
                for c in range(len(t_log_probs)):
                    if c != last_token:
                        new_prefix = prefix + (last_token,) if last_token != self.blank_token_id else prefix
                        new_text = text
                        
                        # Сначала проверяем, нужно ли добавить пробел
                        if c == self.space_token:
                            if text and not text.endswith(' '):  # Добавляем пробел только если текст не пустой и не заканчивается пробелом
                                new_text += " "
                        # Затем добавляем новый символ, если это не пробел и не blank
                        elif c != self.blank_token_id:
                            if c in self.vocab:  # Проверяем наличие символа в словаре
                                new_text += self.vocab[c]
                        
                        new_ac_score = ac_score + t_log_probs[c]
                        new_lm_score = self.lm_model.score(new_text) if new_text else 0.0
                        new_total_score = new_ac_score + self.alpha * new_lm_score + \
                                        self.beta * len(new_text.split())
                        
                        candidates.append((new_prefix, c, new_ac_score, new_lm_score, new_text, new_total_score))
            
            candidates.sort(key=lambda x: x[5], reverse=True)
            beams = candidates[:self.beam_width]
        
        # Финализация лучей
        final_beams = []
        for prefix, last_token, ac_score, lm_score, text, total_score in beams:
            final_text = text
            if last_token != self.blank_token_id and last_token != self.space_token:
                if last_token in self.vocab:
                    final_text += self.vocab[last_token]
            final_beams.append((final_text.strip(), ac_score, lm_score, total_score))
        
        final_beams.sort(key=lambda x: x[3], reverse=True)
        return final_beams[0][0]

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        rescored_beams = []
        for indices, acoustic_score in beams:
            text = self.ctc_decode(indices)
            lm_score = self.lm_model.score(text) if text else 0.0
            total_score = acoustic_score + self.alpha * lm_score
            rescored_beams.append((text, total_score))
        rescored_beams.sort(key=lambda x: x[1], reverse=True)
        
        return rescored_beams[0][0]

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding") 
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")


if __name__ == "__main__":
    
    test_samples = [
        ("samples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("samples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("samples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("samples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("samples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("samples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("samples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("samples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]