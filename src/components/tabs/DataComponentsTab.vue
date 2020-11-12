<template>
  <div>
    <ExpandablePanel name="Custom">
      <ButtonGrid>
        <AddNodeButton node="Custom" name="Custom"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="I/O">
      <ButtonGrid>
        <AddNodeButton node="InData" name="InData"/>
        <AddNodeButton node="OutData" name="OutData"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Transform">
      <ButtonGrid>
        <AddNodeButton node="ToTensor" name="ToTensor"/>
        <AddNodeButton node="Grayscale" name="Grayscale"/>
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
export default class DataComponentsTab extends Vue {
}
</script>
