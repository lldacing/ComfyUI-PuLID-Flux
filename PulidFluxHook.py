import torch
from torch import Tensor
from comfy.ldm.flux.layers import timestep_embedding

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "dit" not in to["patches_replace"]:
        to["patches_replace"]["dit"] = {}
    else:
        to["patches_replace"]["dit"] = to["patches_replace"]["dit"].copy()

    if key not in to["patches_replace"]["dit"]:
        if "double_block" in key:
            to["patches_replace"]["dit"][key] = DitDoubleBlockReplace(pulid_patch, **patch_kwargs)
        else:
            to["patches_replace"]["dit"][key] = DitSingleBlockReplace(pulid_patch, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["dit"][key].add(pulid_patch, **patch_kwargs)

def pulid_patch(img, pulid_model=None, ca_idx=None, weight=1.0, embedding=None, mask=None):
    # apply attn_mask in here?
    pulid_img = weight * pulid_model.pulid_ca[ca_idx].to(img.device)(embedding, img)

    return pulid_img


class DitDoubleBlockReplace:
    def __init__(self, callback, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_args, extra_options):
        sigma = input_args["timestep"][0]
        out = extra_options["original_block"](input_args)
        img = out['img']
        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                img = img + callback(img,
                                     pulid_model=self.kwargs[i]['pulid_model'],
                                     ca_idx=self.kwargs[i]['ca_idx'],
                                     weight=self.kwargs[i]['weight'],
                                     embedding=self.kwargs[i]['embedding'],
                                     mask = self.kwargs[i]['mask'],
                                     )

        out['img'] = img
        return out


class DitSingleBlockReplace:
    def __init__(self, callback, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_args, extra_options):

        out = extra_options["original_block"](input_args)

        sigma = input_args["timestep"][0]
        img = out['img']
        txt = input_args['txt']
        real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                real_img = real_img + callback(real_img,
                                               pulid_model=self.kwargs[i]['pulid_model'],
                                               ca_idx=self.kwargs[i]['ca_idx'],
                                               weight=self.kwargs[i]['weight'],
                                               embedding=self.kwargs[i]['embedding'],
                                               mask=self.kwargs[i]['mask'],
                                               )
        img = torch.cat((txt, real_img), 1)

        out['img'] = img
        return out

def forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control = None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    blocks_replace = patches_replace.get("dit", {})

    for i, block in enumerate(self.double_blocks):
        # 0 -> 18
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       # add timestep for pulid
                                                       "timestep": timesteps,
                                                       "attn_mask": attn_mask},
                                                      {"original_block": block_wrap})
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe,
                             attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add


    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        # 0 -> 37
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       # add txt timestep for pulid
                                                       "txt": txt,
                                                       "timestep": timesteps,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1]:, ...] += add

    img = img[:, txt.shape[1]:, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img