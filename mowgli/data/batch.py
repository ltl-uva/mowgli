

class Batch:
    """Object for holding a batch of parallel data with mask."""
    def __init__(
        self,
        src,        #
        trg,        #
        src_length, #
        trg_length, #
        pad,        # pad index
        src_lang,
        trg_lang,
        graph,
        idxs=None   #
    ):
        self.type = "parallel"
        self.graph = graph
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.idxs = idxs
        self.src = src
        self.src_length = src_length
        self.trg_length = trg_length
        self.src_mask = (self.src != pad).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = trg[:, :-1] # trg_input is used for teacher forcing, last one is cut off
        self.trg = trg[:, 1:] # trg is used for loss computation, shifted by one since BOS
        self.trg_mask = (self.trg_input != pad).unsqueeze(1) # exclude the padded areas from the loss computation
        self.ntokens = (self.trg != pad).data.sum().item()

    def to_device(self, device):
        """Move the parallel batch to `device` (CPU or GPU)."""
        self.src,self.src_mask,self.src_length = self.src.to(device),self.src_mask.to(device),self.src_length.to(device)
        self.trg,self.trg_mask,self.trg_input  = self.trg.to(device),self.trg_mask.to(device),self.trg_input.to(device)