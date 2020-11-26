<template>
  <div>
    Here lives Modal Text
    <UIButton text="Primary" :primary="true"/>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Modal from '@/components/modals/Modal.vue';
import UIButton from '@/components/buttons/UIButton.vue';

@Component({
  components: { UIButton, Modal },
})
export default class ExportModal extends Vue {
  private open = true;
}
</script>

<style>
</style>
