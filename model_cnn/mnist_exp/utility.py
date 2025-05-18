import torch


def img_2_event_img(image, device, spike_ts):
    """
    Transform image to event image

    Args:
        image (Tensor): image
        device (device): device
        spike_ts (int): spike timestep

    Returns:
        event_image: event image

    """
    batch_size = image.shape[0]
    channel_size = image.shape[1]
    image_size = image.shape[2]
    image = image.view(batch_size, channel_size, image_size, image_size, 1)
    random_image = torch.rand([batch_size, channel_size, image_size, image_size, spike_ts], device=device)
    event_image = (random_image < image).float()

    return event_image
