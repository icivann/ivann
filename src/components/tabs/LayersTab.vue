<template>
  <div class="layers-tab">
    <ExpandablePanel name="I/O">
      <ButtonGrid>
        <AddNodeButton node="InModel" name="Input" :names="editorIONames"/>
        <AddNodeButton node="OutModel" name="Output" :names="editorIONames"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Convolutional">
      <ButtonGrid>
        <AddNodeButton node="Conv1d" name="Conv1d"/>
        <AddNodeButton node="Conv2d" name="Conv2d"/>
        <AddNodeButton node="Conv3d" name="Conv3d"/>
        <AddNodeButton node="ConvTranspose1d" name="ConvTranspose1d"/>
        <AddNodeButton node="ConvTranspose2d" name="ConvTranspose2d"/>
        <AddNodeButton node="ConvTranspose3d" name="ConvTranspose3d"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Pooling">
      <ButtonGrid>
        <AddNodeButton node="MaxPool1d" name="MaxPool1d"/>
        <AddNodeButton node="MaxPool2d" name="MaxPool2d"/>
        <AddNodeButton node="MaxPool3d" name="MaxPool3d"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Regularization">
      <ButtonGrid>
        <AddNodeButton node="Dropout" name="Dropout"/>
        <AddNodeButton node="Dropout2d" name="Dropout2d"/>
        <AddNodeButton node="Dropout3d" name="Dropout3d"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Activation Functions">
      <ButtonGrid>
        <AddNodeButton node="Relu" name="Relu"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Custom">
      <ButtonGrid>
        <AddNodeButton node="Custom" name="Custom"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Operations">
      <ButtonGrid>
        <AddNodeButton node="Concat" name="Concat"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Linear">
      <ButtonGrid>
        <AddNodeButton node="Linear" name="Linear"/>
        <AddNodeButton node="Bilinear" name="Bilinear"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Transformer">
      <ButtonGrid>
        <AddNodeButton node="Transformer" name="Transformer"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Non-Linear Activation">
      <ButtonGrid>
        <AddNodeButton node="Softmin" name="Softmin"/>
        <AddNodeButton node="Softmax" name="Softmax"/>
      </ButtonGrid>
    </ExpandablePanel>

  </div>
</template>

<script>
import { Component, Vue } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import { mapGetters } from 'vuex';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
  computed: mapGetters(['editorIONames']),
})
export default class LayersTab extends Vue {
}
</script>
