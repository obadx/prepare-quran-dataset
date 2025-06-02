from typing import Any
from dataclasses import dataclass

import numpy as np
from recitations_segmenter import segment_recitations, clean_speech_intervals


@dataclass
class TruncateOutput:
    audio: list[dict[str, Any]]
    speech_intervals_sec: list[np.ndarray]
    speech_intervals_samples: list[np.ndarray]

    """
    audio: list({'array': np.ndarray, 'sampling_rate': int})
    """


def truncate(
    wav: np.ndarray,
    speech_intervals_sec: np.ndarray,
    sampling_rate=16000,
    truncate_window_overlap_length=16000,
    max_size_samples=480000,
    verbose=False,
) -> TruncateOutput:
    """Moving winodw truncatation arlogrith where the window size is `max_size_samples`
    Note:
    * speech_inatevals are execlusive EX intv = [1, 6] so [1, 2, 3, 4, ,5] are speech
    * speech_intervals are not overlapped
    """

    assert max_size_samples > truncate_window_overlap_length, (
        "`max_size_samples` should be > `truncate_window_overlap_length` "
    )
    speech_intervals_samples = np.array(speech_intervals_sec) * sampling_rate
    speech_intervals_samples = speech_intervals_samples.astype(np.longlong)

    # edge case last interval end should be < total waves length &interval end with inf
    if speech_intervals_samples.shape[0] > 0:
        if speech_intervals_samples[-1][1] > len(wav) or np.isinf(
            speech_intervals_sec[-1][1]
        ):
            speech_intervals_samples[-1][1] = len(wav)

    out = TruncateOutput([], [], [])
    overlap = truncate_window_overlap_length
    window = max_size_samples
    step = window - overlap
    num_items = int(np.ceil(max(0, len(wav) - window) / (window - overlap))) + 1
    if len(wav) == 0:
        num_items = 0

    if verbose:
        print(f"num of items: {num_items}")

    # if verbose:
    #     print(f'before intervals:\n{speech_intervals_samples}')
    #     print(f'before seconds:\n{speech_intervals_sec}')
    #     print(f'len of wav: {len(wav)}')
    #     print(f'num of items: {num_items}')

    start = 0
    intv_start_idx = 0
    for idx in range(num_items):
        end = start + window
        out.audio.append({"array": wav[start:end], "sampling_rate": sampling_rate})

        chosen_idx = intv_start_idx
        frgmented_intv = None
        intv_idx = 0
        for intv_idx in range(intv_start_idx, len(speech_intervals_samples)):
            # print(f' speech_intervals:\n {speech_intervals_samples}')
            # start >= interval end (because of speech iterval end are execlusive)
            if start >= speech_intervals_samples[intv_idx][1]:
                break

            # interval end is smaller than the winodw size
            # ( <=because of speech iterval end are execlusive)
            elif speech_intervals_samples[intv_idx][1] <= end - overlap:
                chosen_idx += 1

            # deviding the speech interval in two parts
            # part to be added to the currect frame(idx)
            # and the other one for the next frame
            elif speech_intervals_samples[intv_idx][0] < end:
                frgmented_intv = np.zeros(2, dtype=np.longlong)
                # in case of overlapping winodws
                frgmented_intv[0] = speech_intervals_samples[intv_idx][0]
                frgmented_intv[1] = min(end, int(speech_intervals_samples[intv_idx][1]))

                # new start for the next iteration
                # if start of speech interval between end and (end -overlap)
                speech_intervals_samples[intv_idx][0] = max(
                    end - overlap, int(speech_intervals_samples[intv_idx][0])
                )
                break

            # TODO: non reachable case
            else:
                break

        if frgmented_intv is None:
            out.speech_intervals_samples.append(
                speech_intervals_samples[intv_start_idx:chosen_idx].copy()
            )
        else:
            out.speech_intervals_samples.append(
                np.concatenate(
                    (
                        speech_intervals_samples[intv_start_idx:chosen_idx].copy(),
                        np.expand_dims(frgmented_intv, 0),
                    ),
                    axis=0,
                ),
            )

        # print('before')
        # print(f'{idx}:\n{np.concatenate(out.speech_intervals_samples, 0)}')
        # print(f'intv idx: {intv_idx}')

        # making intervals relative to each audio frame not the entire audio
        out.speech_intervals_samples[-1] -= start

        # print('after')
        # print(np.concatenate(out.speech_intervals_samples, 0))
        # print('-' * 50)

        # end of the loop
        out.speech_intervals_sec.append(
            out.speech_intervals_samples[-1] / sampling_rate
        )
        start += step
        intv_start_idx = intv_idx

    # if (num_items == 10) and verbose:
    #     print(out.speech_intervals_sec)
    #     print(out.speech_intervals_samples)
    #     print('\n\n\n')

    assert len(out.audio) == len(out.speech_intervals_samples)

    return out
