from abc import ABC, abstractmethod

import jax

__all__ = \
    [
    ]


class PytreeNode(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def flatten(obj):
            return obj.flatten()

        def unflatten(aux_data, children):
            return cls.unflatten(aux_data, children)

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)

    @abstractmethod
    def flatten(self):
        """Return a JAX flattened representation.

        Returns
        -------

        Sequence[Sequence[object, ...], Sequence[object, ...]]
            The JAX flattened representation.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def unflatten(cls, aux_data, children):
        """Unpack a JAX flattened representation.

        Returns
        -------

        object
            The unpacked object.
        """

        raise NotImplementedError
