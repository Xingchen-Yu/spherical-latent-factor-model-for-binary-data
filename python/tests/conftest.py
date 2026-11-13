import sphfactor  # noqa: F401  (sets XLA env defaults before jax import)
import jax

jax.config.update("jax_enable_x64", True)
