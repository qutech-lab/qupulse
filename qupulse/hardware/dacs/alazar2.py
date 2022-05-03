"""This file uses atsaverage.config2"""
import copy
import enum
import logging
from typing import Iterable, Dict, Tuple, Sequence, Mapping
import dataclasses
import numpy
import numpy as np
from numbers import Rational

from qupulse.utils.types import TimeType
from qupulse.hardware.dacs.dac_base import DAC
from qupulse.hardware.dacs.alazar import sample_masks

from atsaverage import config2
from atsaverage.operations import OperationDefinition
from atsaverage.masks import Mask, PeriodicMask, CrossBufferMask

logger = logging.getLogger(__name__)


def try_periodic_mask(identifier: str, begins: np.ndarray, lengths: np.ndarray):
    """Assumes begins is sorted"""
    # TODO: numba

    length = np.unique(lengths)
    if len(length) != 1:
        return None
    else:
        length = length[0]

    period = np.unique(np.diff(begins))
    if len(period) != 1:
        return None
    else:
        period = period[0]

    skip = begins[0] // period
    begin = begins[0] % period
    take = len(begins)
    return PeriodicMask(identifier=identifier,
                        skip=skip, take=take,
                        begin=begin, period=period, end=length - begin)


@dataclasses.dataclass
class AlazarProgram:
    operations: Sequence[OperationDefinition] = dataclasses.field(
        default_factory=list)
    numerical_masks: Mapping[str, Tuple[np.ndarray, np.ndarray]] = dataclasses.field(
        default_factory=dict)
    board_configuration: config2.BoardConfiguration = dataclasses.field(default_factory=config2.BoardConfiguration)
    buffer_strategy_settings: config2.BufferStrategySettings = dataclasses.field(default=None)
    auto_rearm_count: int = dataclasses.field(default=1)

    def get_mask_definitions(self):
        # TODO: optimize
        raise NotImplementedError()


class MaskType(enum.Enum):
    AUTO = 'auto'
    TABLE = 'table'
    PERIODIC = 'periodic'


@dataclasses.dataclass
class MaskPrototype:
    hardware_channel: int
    mask_type: MaskType = dataclasses.field(default=MaskType.AUTO)

    def make_mask(self, identifier: str, begins, lengths) -> Mask:
        mask_type = MaskType(self.mask_type)
        mask = None
        if mask_type in (MaskType.AUTO, MaskType.PERIODIC):
            mask = try_periodic_mask(identifier, begins, lengths)
            if mask is None and mask_type == MaskType.PERIODIC:
                raise ValueError("Cannot make a periodic mask from arguments", begins, lengths)

        if mask is None:
            mask = CrossBufferMask(
                identifier=identifier,
                begin=begins,
                length=lengths
            )

        mask.channel = self.hardware_channel
        return mask


class AlazarQupulse(DAC):
    def __init__(self, ats_card):
        self.ats_card = ats_card
        self.mask_prototypes: Mapping[str, MaskPrototype] = {}

        self.programs: Mapping[str, AlazarProgram] = {}

        self.armed_program = None
        self._remaining_auto_triggers = 0
        self.force_board_config_update = False
        self.default_buffer_strategy = config2.BufferStrategySettings()

    def register_measurement_windows(self, program_name: str,
                                     windows: Dict[str, Tuple[numpy.ndarray,
                                                              numpy.ndarray]]) -> None:
        program = self.programs.setdefault(program_name, AlazarProgram())
        program.numerical_masks = windows

    def set_measurement_mask(self, program_name, mask_name, begins, lengths) -> Tuple[
        numpy.ndarray, numpy.ndarray]:
        program = self.programs.setdefault(program_name, AlazarProgram())
        program.numerical_masks[mask_name] = begins, lengths

    def register_operations(self, program_name: str, operations) -> None:
        program = self.programs.setdefault(program_name, AlazarProgram())
        program.operations = operations

    def arm_program(self, program_name: str) -> None:
        logger.debug("Arming program %s on %r", program_name, self.__card)

        to_arm = self.programs[program_name]
        if self.armed_program != to_arm:
            logger.info("Arming %r by calling applyConfiguration.", self.ats_card)

            # update board config
            self.ats_card.apply_board_configuration(to_arm.board_configuration, force=self.force_board_config_update)
            self.force_board_config_update = False

            sample_rate = self.ats_card.get_current_sample_rate()
            sample_rate_ghz = TimeType.from_fraction(sample_rate, 10**9)

            masks = []
            for mask_id, (begins, lengths) in to_arm.numerical_masks.items():
                # TODO: cache
                begins, lengths = sample_masks(begins, lengths, sample_rate_ghz)

                mask = self.mask_prototypes[mask_id].make_mask(mask_id, begins, lengths)
                masks.append(mask)

            scanline_definition = config2.create_scanline_definition(
                masks,
                to_arm.operations,
                raw_data_mask=0,
                board_spec=self.ats_card.get_board_spec(),
                numeric_sample_rate=sample_rate,
                buffer_strategy=to_arm.buffer_strategy_settings or self.default_buffer_strategy
                            )

            self.ats_card.configureMeasurement(scanline_definition)
            self.armed_program = copy.deepcopy(to_arm)
            self._remaining_auto_triggers = 0

        if self._remaining_auto_triggers == 0:
            self.ats_card.startAcquisition(self.armed_program.auto_rearm_count)
            self._remaining_auto_triggers = self.armed_program.auto_rearm_count - 1

        else:
            self._remaining_auto_triggers -= 1
            logger.info(
                "Relying on atsaverage auto-arm with %d auto triggers remaining after this one",
                self._remaining_auto_triggers)

    def delete_program(self, program_name) -> None:
        del self.programs[program_name]

    def clear(self) -> None:
        self.programs.clear()

    def measure_program(self, channels: Iterable[str]) -> Dict[str, numpy.ndarray]:
        scanline_data = self.ats_card.extractNextScanline()

        scanline_definition = scanline_data.definition
        operation_definitions = {operation.identifier: operation
                                 for operation in scanline_definition.operations}
        mask_definitions = {mask.identifier: mask
                            for mask in scanline_definition.masks}

        def get_input_range(operation_id: str):
            # currently does not work for ComputeMomentDefinition :(
            mask_id = operation_definitions[operation_id].maskID

            hw_channel = int(mask_definitions[mask_id].channel)

            # This fails if new changes have been applied to the card in the meantime
            # It is better than self.config.inputConfiguration but still
            return self.ats_card.board_configuration.input_configuration[hw_channel].inputRange

        data = {}
        for op_name in channels:
            input_range = get_input_range(op_name)
            data[op_name] = scanline_data.operationResults[op_name].getAsVoltage(input_range)

        return data
