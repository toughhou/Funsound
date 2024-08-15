from funsound.utils import *


def greedy_search( log_probs, valid_token_len):
    token_ids = log_probs.argmax(axis=-1)
    token_ids_valid = token_ids[:valid_token_len]
    return token_ids_valid
    

def beam_search_decode(log_probs, beam_size=5, nbest=3):
    """
    Perform beam search decoding with optimizations for large vocabularies.
    
    Args:
        log_probs (numpy.ndarray): TxV matrix of log acoustic posteriors.
        beam_size (int): Number of beams to keep at each step.
        nbest (int): Number of best sequences to return.
    
    Returns:
        list of tuples: List of (sequence, score) tuples, sorted by score.
    """
    T, V = log_probs.shape
    # Initialize the beam with an empty sequence and zero score.
    beam = [(0, [])]
    
    for t in range(T):
        new_beam = []
        for score, seq in beam:
            # Get top beam_size probabilities and their indices
            top_k_indices = np.argpartition(-log_probs[t], beam_size)[:beam_size]
            for v in top_k_indices:
                new_seq = seq + [v]
                new_score = score + log_probs[t, v]
                heapq.heappush(new_beam, (new_score, new_seq))
                # Maintain the size of the heap
                if len(new_beam) > beam_size:
                    heapq.heappop(new_beam)
        # Update beam
        beam = new_beam

    # Sort by score (descending) and return the n-best sequences
    beam.sort(key=lambda x: -x[0])
    return [(seq, score) for score, seq in beam[:nbest]]