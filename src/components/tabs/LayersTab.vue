<template>
  <div class="layers-tab">
    <ExpandablePanel name="Input/Output">
      <ButtonGrid>
        <AddNodeButton node="Input" name="Input" @node-created="inputAdded"/>
        <AddNodeButton node="Output" name="Output" @node-created="outputAdded"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Linear">
      <ButtonGrid>
        <AddNodeButton node="Dense" name="Dense"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Convolutional">
      <ButtonGrid>
        <AddNodeButton node="Convolution1D" name="Conv1D"/>
        <AddNodeButton node="Convolution2D" name="Conv2D"/>
        <AddNodeButton node="Convolution3D" name="Conv3D"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Pooling">
      <ButtonGrid>
        <AddNodeButton node="MaxPooling2D" name="MaxPool2D"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Regularization">
      <ButtonGrid>
        <AddNodeButton node="Dropout" name="Dropout"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Reshaping">
      <ButtonGrid>
        <AddNodeButton node="Flatten" name="Flatten"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Custom">
      <ButtonGrid>
        <AddNodeButton node="Custom" name="Custom"/>
      </ButtonGrid>
    </ExpandablePanel>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
})
export default class LayersTab extends Vue {
  private inputAdded() {
    const { currEditorModel } = this.$store.getters;
    if (currEditorModel !== undefined) {
      const count = currEditorModel.inputs.length;
      const inputName = `Input${count}`;
      currEditorModel.inputs.push({ name: inputName });
    }
  }

  private outputAdded() {
    const { currEditorModel } = this.$store.getters;
    if (currEditorModel !== undefined) {
      const count = currEditorModel.outputs.length;
      const outputName = `Output${count}`;
      currEditorModel.outputs.push({ name: outputName });
    }
  }
}
</script>
