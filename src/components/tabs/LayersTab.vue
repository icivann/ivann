<template>
  <div class="layers-tab">
    <ExpandablePanel name="Linear">
      <ButtonGrid>
        <AddNodeButton node="Convolution2D" name="Conv2D"/>
        <AddNodeButton node="MaxPooling2D" name="MaxPool2D"/>
        <AddNodeButton node="Dropout" name="Dropout"/>
        <AddNodeButton node="Convolution2D" name="Conv2D"/>
        <AddNodeButton node="MaxPooling2D" name="MaxPool2D"/>
        <AddNodeButton node="Dropout" name="Dropout"/>
        <AddNodeButton node="Flatten" name="Flatten"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Convolutional">
      This contains convolution layers. <br/>
      Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
      labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
      laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
      voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
      non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    </ExpandablePanel>
    <ExpandablePanel name="Pooling">
      This contains pooling layer nodes (obvs). <br/>
      Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
      labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
      laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
      voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
      non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    </ExpandablePanel>
  </div>
</template>

<script>
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
}
</script>
