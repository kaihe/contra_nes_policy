"""Vendored OpenAI VPT library subset — the transformer-XL recurrent core.

ROCKET-2 imports ``FanInInitReLULayer`` and ``ResidualRecurrentBlocks`` from
``minestudio.utils.vpt_lib.util``. MineStudio is Minecraft-coupled and not
installed here, so the relevant files are vendored from ROCKET-1's copy of the
same library (``rocket/arm/utils/vpt_lib/``) with imports rewritten. Only
``util.py`` was edited; the rest are verbatim.
"""
