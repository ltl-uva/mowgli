from mowgli.data import Batch

class MultiparallelBatch:
    """Object for holding a multiparallel batch of parallel data with mask."""
    def __init__(
        self,
        src1,
        src2,
        trg,
        src1_length,
        src2_length,
        trg_length,
        pad,
        src1_lang,
        src2_lang,
        trg_lang
    ):
        self.type = "multiparallel"
        self.src1_lang = src1_lang
        self.src2_lang = src2_lang
        self.trg_lang = trg_lang
        self.src1,self.src2 = src1,src2
        self.src1_mask = (self.src1 != pad).unsqueeze(1)
        self.src2_mask = (self.src2 != pad).unsqueeze(1)
        self.src1_length = src1_length
        self.src2_length = src2_length
        self.trg_full = trg
        self.trg_input = trg[:, :-1] # trg_input is used for teacher forcing, last one is cut off
        self.trg = trg[:, 1:] # trg is used for loss computation, shifted by one since BOS
        self.trg_mask = (self.trg_input != pad).unsqueeze(1) # exclude the padded areas from the loss computation
        self.trg_length = trg_length

        # Multiply by two since we have two source sentences and a single corresponding target
        self.ntokens = (self.trg != pad).data.sum().item() * 2
        self.nseqs = self.trg.size(0) * 2
        self.pad = pad

    def to_device(self, device):
        """Moves the multiparallel batch to `device` (CPU or GPU)."""
        self.src1,self.src2 = self.src1.to(device),self.src2.to(device)
        self.src1_mask,self.src2_mask = self.src1_mask.to(device),self.src2_mask.to(device)
        self.trg,self.trg_input,self.trg_mask = self.trg.to(device),self.trg_input.to(device),self.trg_mask.to(device)

    def split_to_parallel(self):
        """Converts the multiparallel batch into two parallel batches."""
        return (
            Batch(self.src1, self.trg_full, self.src1_length, self.trg_length, self.pad, self.src1_lang, self.trg_lang),
            Batch(self.src2, self.trg_full, self.src2_length, self.trg_length, self.pad, self.src2_lang, self.trg_lang),
        )
