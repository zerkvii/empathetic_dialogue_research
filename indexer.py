from BPE.bpe_encoder import TextEncoder


class Indexer:
    def __init__(self, n_ctx=256):
        """
        Including word token indexs, special indexs and postion indexs
        :param n_ctx: max length of model input
        """
        # vocabulary
        self.text_encoder = TextEncoder()
        self.n_vocab = len(self.text_encoder.encoder)  # number of word tokens
        # decoding tokens
        self.SOS_IDX = self.n_vocab + 0
        self.EOS_IDX = self.n_vocab + 1
        # padding index (it doesn't matter what it is, because mask will be used)
        self.PAD_IDX = 0
        # dialog states
        self.DS_SPEAKER_IDX = self.n_vocab + 2
        self.DS_LISTENER_IDX = self.n_vocab + 3
        # emotion label prepender indicator
        self.DS_PREP_IDX = self.n_vocab + 4
        # the numbers of special indexs (SOS, EOS, DS_S, DS_L, DS_PREP), for input embedding
        self.n_special = 5
        # position indexs
        self.n_ctx = n_ctx

    def encode_text(self, x):
        # from text to indexs
        return self.text_encoder.encode(x)

    def decode_index2text(self, idx):
        # from indexs to text (indexs range in [0,n_vocab+2) (SOS, EOS))
        if idx == self.SOS_IDX:
            return '[SOS]'
        elif idx == self.EOS_IDX:
            return '[EOS]'
        elif idx==self.DS_PREP_IDX:
            return '</pre_emo>'
        elif idx==self.DS_LISTENER_IDX:
            return '</listener>'
        elif idx==self.DS_SPEAKER_IDX:
            return'</speaker>'
        else:
            return self.text_encoder.decoder[idx]
    def decode_text_sequence(self,id_seq_tensor):
        seq_lis=[]
        for idx in id_seq_tensor:
            seq_lis.append(self.decode_index2text(idx.item()))
        return ' '.join(seq_lis).replace('</w>','')