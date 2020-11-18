<template>
  <div class="search-tab h-100">
    <ExpandablePanel
      v-for="(file) in files"
      :key="file.filename"
      :name="file.filename"
    >
      <ButtonGrid>
        <AddNodeButton
          v-for="(func) in file.functions"
          :key="func.name"
          :node="customNode"
          :name="func.name"
          :options="func"
        />
      </ButtonGrid>
    </ExpandablePanel>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { mapGetters } from 'vuex';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import { CommonNodes } from '@/nodes/common/Types';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
  computed: mapGetters(['files']),
})
export default class CustomTab extends Vue {
  private customNode: string = CommonNodes.Custom;
}
</script>

<style scoped>
</style>
