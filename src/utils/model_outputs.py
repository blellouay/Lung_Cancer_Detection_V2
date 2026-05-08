def get_main_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits

    return outputs


def get_aux_logits(outputs):
    if hasattr(outputs, "aux_logits"):
        return outputs.aux_logits

    return None


def classification_loss(outputs, labels, criterion, aux_weight=0.4):
    logits = get_main_logits(outputs)
    loss = criterion(logits, labels)

    aux_logits = get_aux_logits(outputs)
    if aux_logits is not None:
        loss = loss + aux_weight * criterion(aux_logits, labels)

    return loss
